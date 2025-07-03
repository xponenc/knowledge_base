import time
import traceback
from typing import List, Dict, Tuple, Optional
from celery_progress.backend import ProgressRecorder
from celery import shared_task
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os

from django.core.files.base import ContentFile
from django.db.models import Prefetch, OuterRef, Subquery
from django.core.files import File
from django.db.models import Prefetch

from app_sources.content_models import RawContent, ContentStatus, CleanedContent
from app_sources.report_models import CloudStorageUpdateReport, ReportStatus
from app_sources.services.content_process import recognize_and_summarize_content
from app_sources.services.summary import summarize_text
from app_sources.source_models import NetworkDocument, SourceStatus
from app_sources.storage_models import CloudStorage
from app_tasks.models import ContentComparison, TaskForSource
from utils.process_files import compute_sha512

logger = logging.getLogger(__name__)
document_logger = logging.getLogger("document_processing")


@shared_task(bind=True)
def download_and_create_raw_content_parallel_task(self,
                                                  document_ids: List[int],
                                                  update_report_pk: int,
                                                  max_workers: int = 5):
    """
    Celery-задача для параллельной загрузки и обработки файлов из облачного хранилища.

    :param self: Celery task instance.
    :param document_ids: Список ID NetworkDocument для обработки.
    :param update_report_pk: PK объекта CloudStorageUpdateReport.
    :param max_workers: Максимальное количество потоков.
    """
    storage_update_report = CloudStorageUpdateReport.objects.select_related("storage", "author").get(pk=update_report_pk)
    cloud_storage = storage_update_report.storage
    author = storage_update_report.author

    # Получаем документы и текущий raw_content (для сравнения)
    raw_qs = RawContent.objects.filter(status=ContentStatus.READY.value).order_by("-created_at")[:1]
    documents = NetworkDocument.objects.filter(pk__in=document_ids).prefetch_related(
        Prefetch('rawcontent_set', queryset=raw_qs, to_attr='related_rawcontents'))

    for doc in documents:
        doc.current_raw_content = doc.related_rawcontents[0] if doc.related_rawcontents else None

    total = len(documents)
    if total == 0:
        return "Документы не найдены."

    # Прогресс-бар
    progress_recorder = ProgressRecorder(self)
    progress_recorder.set_progress(0, total, description="Загрузка контента из хранилища")

    success, failed = [], []

    # Параллельная загрузка
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_and_process_file, doc, cloud_storage, storage_update_report, author): doc.pk
            for doc in documents
        }

        for i, future in enumerate(as_completed(futures)):
            try:
                pk, status = future.result()
                if status == "success":
                    success.append(pk)
                else:
                    failed.append(pk)
            except Exception as e:
                logger.exception(f"Непредвиденная ошибка при обработке документа: {e}")
                failed.append(futures[future])
            finally:
                progress_recorder.set_progress(i + 1, total, description="Загрузка и сохранение контента")

    # Итоговое логгирование
    document_logger.info(f"Загрузка завершена. Успешно: {len(success)}, Ошибки: {len(failed)}")
    storage_update_report.content.setdefault("processing", {})["raw_content"] = {
        "success": success,
        "failed": failed
    }
    storage_update_report.save(update_fields=["content"])
    return f"Готово: success={len(success)}, failed={len(failed)}"


def download_and_process_file(doc: NetworkDocument,
                               cloud_storage,
                               storage_update_report: CloudStorageUpdateReport,
                               author) -> Tuple[int, str]:
    """
    Загружает файл из облака, создаёт RawContent, сравнивает хеши, вызывает TaskForSource при необходимости.

    :param doc: Документ.
    :param cloud_storage: Объект облачного хранилища.
    :param storage_update_report: Объект отчёта обновления.
    :param author: Пользователь, инициировавший обновление.
    :return: Кортеж (pk, 'success'|'fail').
    """
    temp_path, file_name = None, None
    logger.info(f"[Doc {doc.pk}] Загрузка файла: {doc.url}")

    try:
        cloud_api = cloud_storage.get_storage()
        temp_path, file_name = cloud_api.download_file_to_disk_sync(doc.url)
        logger.info(f"[Doc {doc.pk}] Файл загружен: {temp_path}")
    except Exception as e:
        msg = f"Ошибка загрузки: {e}"
        logger.error(f"[Doc {doc.pk}] {msg}")
        storage_update_report.content.setdefault("errors", []).append(f"[{doc.pk}] {msg}")
        storage_update_report.status = "error"
        return doc.pk, "fail"

    try:
        raw_content = RawContent(
            network_document=doc,
            report=storage_update_report,
            author=author
        )
        with open(temp_path, "rb") as f:
            raw_content.file.save(file_name, File(f), save=False)

        raw_content.hash_content = compute_sha512(temp_path)
        raw_content.save()

        logger.info(f"[Doc {doc.pk}] RawContent сохранён.")

        # Если есть старый RawContent — сравниваем и создаём задачу
        if doc.status != SourceStatus.CREATED.value:
            try:
                create_task_for_network_document(
                    doc=doc,
                    storage_update_report=storage_update_report,
                    current_raw_content=doc.current_raw_content,
                    new_raw_content=raw_content
                )
            except Exception as e:
                raw_content.status = ContentStatus.ERROR.value
                raw_content.error_message = f"Ошибка создания задачи: {e}"
                raw_content.save()
                logger.error(f"[Doc {doc.pk}] {raw_content.error_message}")
        else:
            doc.status = SourceStatus.READY.value
            doc.save(update_fields=["status"])

    except Exception as e:
        msg = f"Ошибка обработки файла: {e}"
        logger.error(f"[Doc {doc.pk}] {msg}")
        storage_update_report.content.setdefault("errors", []).append(f"[{doc.pk}] {msg}")
        storage_update_report.status = "error"
        return doc.pk, "fail"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug(f"[Doc {doc.pk}] Временный файл удалён: {temp_path}")

    return doc.pk, "success"


def safe_remove_file(path: str, retries: int = 3, delay: float = 0.5):
    """
    Безопасное удаление файла с несколькими попытками.

    :param path: Путь к файлу для удаления.
    :param retries: Количество попыток удаления.
    :param delay: Задержка между попытками в секундах.
    :return: True, если файл успешно удалён, иначе False.
    """
    for attempt in range(1, retries + 1):
        try:
            if os.path.exists(path) and os.path.isfile(path):
                os.remove(path)
                logger.debug(f"Временный файл удалён: {path}")
                return True
            else:
                logger.debug(f"Файл для удаления не найден или не является файлом: {path}")
                return False
        except Exception as e:
            logger.warning(f"Попытка {attempt} удаления файла {path} не удалась: {e}", exc_info=True)
            time.sleep(delay)
    logger.error(f"Не удалось удалить файл {path} после {retries} попыток")
    return False


def get_documents_for_redownload(
        documents: List[NetworkDocument],
        cloud_storage: CloudStorage,
        progress_recorder: ProgressRecorder = None,
        total: int = None,
) -> List[Tuple[int, Optional[str], Optional[str]]]:
    """
    Определяет документы, для которых необходимо повторно загрузить контент,
    проверяя хеши файлов в облачном хранилище.

    :param documents: Список NetworkDocument для проверки.
    :param cloud_storage: Объект облачного хранилища с методом download_file_to_disk_sync.
    :param progress_recorder: Объект ProgressRecorder для обновления прогресса.
    :param total: Общее количество документов для прогресса (если None, вычисляется автоматически).
    :return: Список кортежей (id документа, путь к временному файлу или None, имя файла или None).
    """
    docs_to_download = []
    total = total or len(documents)

    for i, doc in enumerate(documents, 1):
        temp_path = None
        try:
            temp_path, file_name = cloud_storage.get_storage().download_file_to_disk_sync(doc.url)
            current_hash = compute_sha512(temp_path)

            if doc.last_hash != current_hash:
                logger.info(f"NetworkDocument [id {doc.pk}] Хеш изменился, файл будет перезаписан")
                docs_to_download.append((doc.id, temp_path, file_name))
                temp_path = None  # передаем владельцу списка, не удаляем здесь
            else:
                logger.info(f"NetworkDocument [id {doc.pk}] Хеш совпадает, файл пропускается")

        except Exception as e:
            logger.warning(f"NetworkDocument [id {doc.pk}] Ошибка при проверке хеша: {e}", exc_info=True)
            docs_to_download.append((doc.id, None, None))

        finally:
            if temp_path:
                safe_remove_file(temp_path)

        # Обновление прогресса каждые 10 документов
        if progress_recorder and i % 10 == 0:
            progress_recorder.set_progress(i, total, description="Проверка документов на изменение")

    return docs_to_download


def create_task_for_network_document(doc: NetworkDocument,
                                     storage_update_report: CloudStorageUpdateReport,
                                     current_raw_content: RawContent,
                                     new_raw_content: RawContent):
    """
    Создание задачи TaskForSource на изменение контента, если статус документа != CREATED.

    :param doc: NetworkDocument
    :param storage_update_report: CloudStorageUpdateReport
    :param current_raw_content: старый RawContent
    :param new_raw_content: новый RawContent
    """
    comparison = ContentComparison.objects.create(
        content_type="raw_content",
        old_raw_content=current_raw_content,
        new_raw_content=new_raw_content
    )

    task = TaskForSource(
        cloud_report=storage_update_report,
        network_document=doc,
        comparison=comparison,
        source_previous_status=doc.status
    )

    descriptions = {
        SourceStatus.READY.value: "Обновился исходный контент для АКТИВНОГО источника...",
        SourceStatus.DELETED.value: "Обновился исходный контент для УДАЛЕННОГО источника...",
        SourceStatus.EXCLUDED.value: "Обновился исходный контент для ИСКЛЮЧЕННОГО источника...",
        SourceStatus.ERROR.value: "Обновился исходный контент для источника в статусе ОШИБКА...",
        SourceStatus.WAIT.value: "Обновился исходный контент для источника в статусе ОБРАБОТКА...",
    }
    task.description = descriptions.get(doc.status, "")

    if task.description:
        task.save()
    else:
        logger.error(f"[Doc {doc.pk}] Не удалось определить описание для статуса {doc.status}")




@shared_task(bind=True)
def process_cloud_files(
    self,
    files: List[Dict],
    update_report_pk: int,
    recognize_content: bool = False,
    do_summarization: bool = False

) -> str:
    """
    Основная задача для последовательной обработки файлов в облачном хранилище
    с прогрессом и параллельной загрузкой новых/обновлённых файлов.

    :param self: объект задачи Celery (для прогресса).
    :param files: список словарей с файлами (для bulk-режима), либо пустой список для полной обработки.
    :param update_report_pk: PK отчёта обновления.
    :param recognize_content: запуск OCR.
    :param do_summarization: запуск суммаризации.
    :return: строка результата.
    """
    storage_update_report = get_update_report(update_report_pk)
    cloud_storage_api = initialize_cloud_storage_api(storage_update_report)
    synchronization_type, db_documents, storage_files = fetch_storage_files(files, cloud_storage_api, storage_update_report)

    classification_result = classify_documents(
        synchronization_type,
        db_documents,
        storage_files,
        storage_update_report,
        self,
    )

    created_ids = create_new_documents(classification_result.get("new_files", []), storage_update_report, self, cloud_storage_api)
    updated_ids = process_existing_documents(classification_result.get("exist_documents", []), storage_update_report, self, cloud_storage_api)

    # OCR и суммаризация — делаем в конце для всех новых и обновленных документов
    final_recognition_and_summary(
        storage_update_report,
        recognize_content,
        do_summarization,
        self,
        created_ids + updated_ids,
    )

    update_report_with_results(storage_update_report, classification_result)

    return "Обработка завершена"


def get_update_report(update_report_pk: int):
    """
    Получить объект отчёта обновления по PK.

    :param update_report_pk: PK отчёта.
    :return: CloudStorageUpdateReport объект.
    """
    return CloudStorageUpdateReport.objects.select_related("storage", "author").get(pk=update_report_pk)


def initialize_cloud_storage_api(storage_update_report):
    """
    Инициализация API облачного хранилища из объекта отчёта.

    :param storage_update_report: объект отчёта обновления.
    :return: инициализированный API хранилища.
    :raises: ValueError при ошибке инициализации.
    """
    cloud_storage = storage_update_report.storage
    try:
        api = cloud_storage.get_storage()
        logger.info(f"API хранилища {cloud_storage.name} успешно инициализировано")
        return api
    except ValueError as e:
        logger.error(f"Ошибка инициализации API хранилища {cloud_storage.name}: {e}")
        storage_update_report.content.setdefault("errors", []).append(str(e))
        storage_update_report.status = ReportStatus.ERROR.value
        storage_update_report.save()
        raise


def fetch_storage_files(files, cloud_storage_api, storage_update_report):
    """
    Получить файлы из облачного хранилища и документы из базы.

    :param files: список файлов (bulk), или пустой список для полной синхронизации.
    :param cloud_storage_api: API облачного хранилища.
    :param storage_update_report: объект отчёта.
    :return: tuple (synchronization_type:str, db_documents: QuerySet, storage_files: list[dict])
    """
    cloud_storage = storage_update_report.storage
    if files:
        synchronization_type = "bulk"
        storage_update_report.content["type"] = "bulk"
        db_documents = NetworkDocument.objects.filter(storage=cloud_storage, pk__in=files)
        storage_files = None  # TODO: при bulk можно оптимизировать, сейчас None
    else:
        synchronization_type = "all"
        storage_update_report.content["type"] = "all"
        db_documents = NetworkDocument.objects.filter(storage=cloud_storage)
        storage_files = cloud_storage_api.list_directory(path=cloud_storage_api.root_path)
    storage_update_report.save(update_fields=["content"])
    return synchronization_type, db_documents, storage_files


def classify_documents(synchronization_type, db_documents, storage_files, storage_update_report, task_self):
    """
    Классифицирует файлы на новые, существующие, удалённые.

    :param synchronization_type: "bulk" или "all"
    :param db_documents: QuerySet документов из БД.
    :param storage_files: список файлов из хранилища.
    :param storage_update_report: объект отчёта.
    :param task_self: текущая задача Celery (для прогресса).
    :return: dict с классификацией файлов.
    """
    total_counter = len(storage_files) if storage_files else 0
    if total_counter == 0:
        return {}

    progress_recorder = ProgressRecorder(task_self)
    progress_description = f'Классификация {total_counter} файлов'
    progress_recorder.set_progress(0, total_counter, description=progress_description)

    result = {
        'new_files': [],
        'exist_documents': [],
        'skipped_documents': [],
        'updated_documents': [],
        'error': None,
        'deleted_files': [],
    }

    db_documents_by_url = {doc.url: doc for doc in db_documents}
    incoming_urls_set = set(file["url"] for file in storage_files) if storage_files else set()

    if synchronization_type == "all":
        deleted_urls_set = set(db_documents_by_url) - incoming_urls_set
        deleted_docs = db_documents.filter(url__in=deleted_urls_set)
        for doc in deleted_docs:
            result["deleted_files"].append({
                "url": doc.url,
                "name": getattr(doc, "title", ""),
                "status": doc.status,
            })

    for index, file in enumerate(storage_files or []):
        url = file.get('url')
        if url not in db_documents_by_url:
            result['new_files'].append(file)
        else:
            result['exist_documents'].append(db_documents_by_url[url].pk)

        progress_recorder.set_progress(index + 1, total_counter, description=progress_description)

    storage_update_report.content["result"] = result
    storage_update_report.save(update_fields=["content"])

    return result


def create_new_documents(
    new_files: List[Dict],
    storage_update_report,
    task_self,
    cloud_storage_api,
) -> List[int]:
    """
    Создать новые NetworkDocument и загрузить их RawContent параллельно с прогрессом.

    :param new_files: список новых файлов.
    :param storage_update_report: объект отчёта.
    :param task_self: задача Celery (для прогресса).
    :param cloud_storage_api: API хранилища.
    :return: список ID созданных документов.
    """
    created_ids = []
    if not new_files:
        return created_ids

    cloud_storage = storage_update_report.storage
    batch_size = 500
    bulk_container = []

    for file_data in new_files:
        bulk_container.append(NetworkDocument(
            storage=cloud_storage,
            report=storage_update_report,
            title=file_data["file_name"],
            path=file_data["path"],
            file_id=file_data["file_id"],
            url=file_data["url"],
        ))
        if len(bulk_container) >= batch_size:
            created_docs = NetworkDocument.objects.bulk_create(bulk_container)
            created_ids.extend([doc.id for doc in created_docs])
            bulk_container.clear()

    if bulk_container:
        created_docs = NetworkDocument.objects.bulk_create(bulk_container)
        created_ids.extend([doc.id for doc in created_docs])

    # Параллельно загрузим RawContent для новых документов
    if created_ids:
        process_and_attach_raw_content(
            task_self,
            document_ids=created_ids,
            storage_update_report=storage_update_report,
            cloud_storage_api=cloud_storage_api,
        )

    return created_ids


def process_existing_documents(
    exist_document_ids: List[int],
    storage_update_report,
    task_self,
    cloud_storage_api,
) -> List[int]:
    """
    Проверить и обновить контент существующих документов.

    :param exist_document_ids: список PK существующих документов.
    :param storage_update_report: объект отчёта.
    :param task_self: задача Celery (для прогресса).
    :param cloud_storage_api: API хранилища.
    :return: список обновлённых ID документов.
    """
    updated_doc_ids = []
    if not exist_document_ids:
        return updated_doc_ids

    cloud_storage = storage_update_report.storage

    # Получаем хеши последнего RawContent для сравнения
    docs_to_check = NetworkDocument.objects.filter(pk__in=exist_document_ids).annotate(
        last_hash=Subquery(
            RawContent.objects.filter(
                network_document=OuterRef("pk"),
                status=ContentStatus.READY.value,
            ).order_by("-created_at").values("hash_content")[:1]
        )
    )

    progress_recorder = ProgressRecorder(task_self)
    progress_description = f"Обновление контента {len(exist_document_ids)} документов"
    progress_recorder.set_progress(0, len(exist_document_ids), description=progress_description)

    # Получаем документы, требующие загрузки новых RawContent
    docs_to_download = get_documents_for_redownload(
        documents=docs_to_check,
        cloud_storage=cloud_storage,
        progress_recorder=progress_recorder,
    )

    updated_doc_ids.extend([item[0] for item in docs_to_download])
    updated_docs_with_files = [item for item in docs_to_download if item[1] is not None]
    updated_docs_without_files = [item for item in docs_to_download if item[1] is None]

    # Обрабатываем обновления с готовыми файлами
    raw_content_qs = RawContent.objects.filter(
        status=ContentStatus.READY.value
    ).order_by("-created_at")[:1]

    docs_to_update = NetworkDocument.objects.filter(pk__in=updated_doc_ids).prefetch_related(
        Prefetch('rawcontent_set', queryset=raw_content_qs, to_attr='related_rawcontents'))

    for doc in docs_to_update:
        doc.current_raw_content = doc.related_rawcontents[0] if doc.related_rawcontents else None

    docs_map = {doc.id: doc for doc in docs_to_update}

    for doc_id, temp_path, file_name in updated_docs_with_files:
        try:
            doc = docs_map.get(doc_id)
            current_raw_content = doc.current_raw_content

            new_raw_content = RawContent(
                network_document=doc,
                report=storage_update_report,
                author=storage_update_report.author,
            )
            with open(temp_path, "rb") as f:
                new_raw_content.file.save(file_name, File(f), save=False)
            new_raw_content.hash_content = compute_sha512(temp_path)
            new_raw_content.save()
            logger.info(f"[Document {doc.pk}] RawContent обновлён из кэша")

            # Создаём задачу на изменения
            try:
                create_task_for_network_document(
                    doc=doc,
                    storage_update_report=storage_update_report,
                    current_raw_content=current_raw_content,
                    new_raw_content=new_raw_content,
                )
            except Exception as e:
                logger.error(f"Ошибка создания задачи изменения NetworkDocument [id {doc.pk}]: {e}")
                new_raw_content.status = ContentStatus.ERROR.value
                new_raw_content.error_message = f"Ошибка создания задачи изменения: {e}"
                new_raw_content.save()

        except Exception as e:
            storage_update_report.content.setdefault("errors", []).append(
                f"[Document {doc_id}] Ошибка при создании RawContent из temp_path: {e}")
            storage_update_report.status = ReportStatus.ERROR.value
            logger.error(f"[Document {doc_id}] Ошибка при создании RawContent из temp_path: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # Для документов без готовых файлов запускаем параллельную загрузку
    if updated_docs_without_files:
        document_ids_to_dl = [doc_id for doc_id, _, _ in updated_docs_without_files]
        process_and_attach_raw_content(
            task_self,
            document_ids=document_ids_to_dl,
            storage_update_report=storage_update_report,
            cloud_storage_api=cloud_storage_api,
        )

    return updated_doc_ids


def process_and_attach_raw_content(
    task_self,
    document_ids: List[int],
    storage_update_report,
    cloud_storage_api,
    max_workers: int = 5,
):
    """
    Параллельная загрузка и создание RawContent для документов с прогрессом.

    :param task_self: объект задачи Celery (для прогресса).
    :param document_ids: список ID документов для обработки.
    :param storage_update_report: объект отчёта обновления.
    :param cloud_storage_api: API облачного хранилища.
    :param max_workers: макс. количество потоков.
    """
    if not document_ids:
        return

    documents = NetworkDocument.objects.filter(pk__in=document_ids).prefetch_related(
        Prefetch(
            'rawcontent_set',
            queryset=RawContent.objects.filter(status=ContentStatus.READY.value).order_by("-created_at")[:1],
            to_attr='related_rawcontents',
        )
    )

    for doc in documents:
        doc.current_raw_content = doc.related_rawcontents[0] if doc.related_rawcontents else None

    total = len(documents)
    progress_recorder = ProgressRecorder(task_self)
    progress_description_base = f"Загрузка и создание RawContent для {total} документов"
    progress_recorder.set_progress(0, total, description=progress_description_base)

    current = 0
    results = []

    def _download_and_process(doc):
        try:
            # Вызов функции загрузки и обработки файла
            return download_and_process_file(doc, cloud_storage_api, storage_update_report,
                                             storage_update_report.author)
        except Exception as e:
            logger.exception(f"Ошибка обработки документа {doc.pk}: {e}")
            return (doc.pk, "fail")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_and_process, doc): doc.pk for doc in documents}

        for future in as_completed(futures):
            current += 1
            try:
                result = future.result()
            except Exception as e:
                logger.exception(f"Ошибка в потоке при обработке документа: {e}")
                result = (None, "fail")
            results.append(result)

            # Вычисляем прогресс в процентах для более информативного описания
            percent = int(current / total * 100)
            description = f"{progress_description_base} ({percent}%)"

            # Обновляем прогресс с текущим значением и описанием с %
            progress_recorder.set_progress(current, total, description=description)

    success_ids = [doc_id for doc_id, status in results if status == "success"]
    failed_ids = [doc_id for doc_id, status in results if status == "fail"]

    # Логируем успешные и неудачные
    logger.info(f"Успешно обработано {len(success_ids)} из {total} документов")
    if failed_ids:
        logger.warning(f"Не удалось обработать документы: {failed_ids}")

    return results


def download_and_process_file(
        doc,
        cloud_storage_api,
        storage_update_report,
        author
) -> Tuple[Optional[int], str]:
    """
    Скачивает файл, создаёт RawContent и создаёт задачи для NetworkDocument.

    :param doc: NetworkDocument объект.
    :param cloud_storage_api: API облачного хранилища.
    :param storage_update_report: объект отчёта.
    :param author: пользователь для RawContent.author.
    :return: tuple (document_id, "success"/"fail")
    """
    temp_path = None
    file_name = None

    logger.info(f"[Document {doc.pk}] Начинаем обработку URL: {doc.url}")

    try:
        temp_path, file_name = cloud_storage_api.download_file_to_disk_sync(doc.url)
        logger.info(f"[Document {doc.pk}] Файл скачан во временное хранилище: {temp_path}")
    except Exception as e:
        storage_update_report.content.setdefault("errors", []).append(
            f"[Document {doc.pk}] Ошибка при загрузке: {e}")
        storage_update_report.status = ReportStatus.ERROR.value
        logger.error(f"[Document {doc.pk}] Ошибка при загрузке: {e}")
        return doc.pk, "fail"

    try:
        raw_content = RawContent.objects.create(
            network_document=doc,
            report=storage_update_report,
            author=author,
        )
        with open(temp_path, 'rb') as f:
            raw_content.file.save(file_name, File(f), save=False)

        raw_content.hash_content = compute_sha512(temp_path)
        raw_content.save()

        logger.info(f"[Document {doc.pk}] RawContent создан и сохранён")

        if doc.status != SourceStatus.CREATED.value:
            try:
                create_task_for_network_document(
                    doc=doc,
                    storage_update_report=storage_update_report,
                    current_raw_content=doc.current_raw_content,
                    new_raw_content=raw_content,
                )
            except Exception as e:
                logger.error(f"Ошибка создания задачи для NetworkDocument [id {doc.pk}]: {e}")
                raw_content.status = ContentStatus.ERROR.value
                raw_content.error_message = f"Ошибка создания задачи: {e}"
                raw_content.save()
        else:
            doc.status = SourceStatus.READY.value
            doc.save(update_fields=["status"])

    except Exception as e:
        storage_update_report.content.setdefault("errors", []).append(
            f"[Document {doc.pk}] Ошибка при сохранении RawContent: {e}")
        storage_update_report.status = ReportStatus.ERROR.value
        logger.error(f"[Document {doc.pk}] Ошибка при сохранении RawContent: {e}")
        return doc.pk, "fail"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug(f"[Document {doc.pk}] Временный файл удалён: {temp_path}")

    return doc.pk, "success"


def final_recognition_and_summary(
    storage_update_report,
    recognize_content: bool,
    do_summarization: bool,
    task_self,
    processed_doc_ids: List[int]
):
    """
    Выполняет OCR и суммаризацию для готовых RawContent документов.

    :param storage_update_report: объект отчёта.
    :param recognize_content: флаг OCR.
    :param do_summarization: флаг суммаризации.
    :param task_self: задача Celery.
    :param processed_doc_ids: ID документов.
    """
    if not recognize_content or not processed_doc_ids:
        return

    raw_contents = RawContent.objects.filter(
        network_document__in=processed_doc_ids,
        status=ContentStatus.READY.value
    ).select_related("network_document")

    total = raw_contents.count()
    if total == 0:
        return

    progress_recorder = ProgressRecorder(task_self)
    total = len(raw_contents)
    progress_description_base = f"Распознавание и суммаризация ({total} файлов)"
    progress_step = max(1, total // 100)
    progress_recorder.set_progress(0, total, description=progress_description_base)

    for i, raw_content in enumerate(raw_contents):
        try:
            recognize_and_summarize_content(
                raw_content=raw_content,
                user_id=storage_update_report.author_id,
                do_summarization=do_summarization
            )
        except Exception as e:
            document_logger.error(f"Ошибка при распознавании RawContent {raw_content.id}: {e}")
            raw_content.status = ContentStatus.ERROR.value
            raw_content.error_message = str(e)
            raw_content.save()

        # Вычисляем прогресс в процентах для обновления описания
        percent = int((i + 1) / total * 100)
        description = f"{progress_description_base} ({percent}%)"

        # Обновляем прогресс при достижении шагов и в конце
        if (i + 1) % progress_step == 0 or (i + 1) == total:
            progress_recorder.set_progress(i + 1, total, description=description)

    progress_recorder.set_progress(total, total, description="Распознавание завершено")


def update_report_with_results(storage_update_report, result):
    """
    Обновляет отчёт с результатами.

    :param storage_update_report: объект отчёта.
    :param result: словарь с классификацией.
    """
    storage_update_report.content["result"] = result
    storage_update_report.save(update_fields=["content"])






@shared_task(bind=True)
def process_raw_content_task(self, recognizer_class, raw_content_id, user_id=None):
    """Распознавание контента и запись саммари"""
    progress_recorder = ProgressRecorder(self)

    progress_description = f'Распознается объект'
    progress_recorder.set_progress(20, 100, description=progress_description)
    try:
        raw_content = RawContent.objects.get(pk=raw_content_id)
        # dispatcher = ContentRecognizerDispatcher()
        #
        # file_extension = raw_content.file_extension()
        # recognizers = dispatcher.get_recognizers_for_extension(file_extension)
        #
        # if not recognizers:
        #     raise Exception("Нет распознавателей для этого формата")
        #
        # recognizer_class = recognizers[0]
        recognizer = recognizer_class(raw_content.file.path)
        recognizer_report = recognizer.recognize()

        recognized_text = recognizer_report.get("text", "")
        recognition_method = recognizer_report.get("method", "")
        recognition_quality_report = recognizer_report.get("quality_report", {})

        # if not recognized_text.strip():
        #     raise ValueError("Не удалось распознать текст.")
        # recognized_text = recognizer_report.get("text", "")
        if not recognized_text or not recognized_text.strip():
            raise ValueError("Не удалось распознать текст.")
        progress_description = f'Объект распознан'
        progress_recorder.set_progress(40, 100, description=progress_description)

        # Удаляем старое
        CleanedContent.objects.filter(raw_content=raw_content).delete()

        # Создаем новый CleanedContent
        cleaned_content = CleanedContent.objects.create(
            network_document=raw_content.network_document,
            raw_content=raw_content,
            recognition_method=recognition_method,
            recognition_quality=recognition_quality_report,
            preview=recognized_text[:200],
            author_id=user_id
        )
        # Сохраняем summary как файл (если нужно)
        cleaned_content.file.save("cleaned.txt", ContentFile(recognized_text.encode("utf-8")))
        cleaned_content.save()
        progress_description = f'Выполняем саммаризацию'
        progress_recorder.set_progress(60, 100, description=progress_description)

        # Суммаризация
        # summary = summarize_with_sber(recognized_text)
        summary = summarize_text(recognized_text, mode='summary big')
        if summary is not None:
            document = raw_content.network_document
            document.description = summary
            document.save(update_fields=["description", ])
        progress_description = f'Выполняем саммаризацию'
        progress_recorder.set_progress(100, 100, description=progress_description)
        return "Операция завершена"

    except Exception as e:
        # return {"status": "error", "message": str(e), "trace": traceback.format_exc()}
        return f"Ошибка {traceback.format_exc()}"