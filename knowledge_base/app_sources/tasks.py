import asyncio
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from math import ceil
from pprint import pprint

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.core.files import File
from django.shortcuts import get_object_or_404
from django.utils import timezone
from dateutil.parser import parse

from app_sources.content_models import ContentStatus, RawContent
from app_sources.report_model import CloudStorageUpdateReport, ReportStatus
from app_sources.source_models import NetworkDocument
from app_sources.storage_models import CloudStorage
from utils.process_files import compute_sha512
from django.contrib.auth import get_user_model

User = get_user_model()

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def process_cloud_files(
    self,
    files: list[dict],
    cloud_storage_pk: int,
    update_report_pk: int,
    author_pk: int
):
    """
    Синхронизирует список файлов из облачного хранилища с локальной базой данных.

    Производится предварительная категоризация файлов для последующей обработки:
    - `new_files`: файлы, которых ещё нет в БД
    - `updated_files`: файлы, уже существующие в БД и не имеющие особого статуса
    - `deleted_files`: файлы, которые были в БД, но больше не существуют в облаке
    - `restored_files`: файлы, ранее помеченные как DELETED, но снова появившиеся в облаке
    - `excluded_files`: файлы со статусом EXCLUDED, которые есть в облаке

    :param update_report_pk: id объекта отчета CloudStorageUpdateReport для хранения результата синхронизации
    :param synchronization_type: тип синхронизации 'bulk' частичный по списку, 'all' - полный по хранилищу
    :param self: Контекст Celery-задачи
    :param files: Список словарей с метаинформацией по файлам из облачного хранилища
    :param cloud_storage: Объект CloudStorage, с которым идёт синхронизация
    :return: Строка статуса завершения
    :raises: None
    """
    cloud_storage = get_object_or_404(CloudStorage, pk=cloud_storage_pk)
    storage_update_report = CloudStorageUpdateReport.objects.get(pk=update_report_pk)

    try:
        cloud_storage_api = cloud_storage.get_storage()
        logger.info(f"API хранилища {cloud_storage.name} успешно инициализировано")
    except ValueError as e:
        logger.error(f"Ошибка инициализации API хранилища {cloud_storage.name}: {e}")
        storage_update_report.content.setdefault("errors", []).append(str(e))
        storage_update_report.status = ReportStatus.ERROR.value
        storage_update_report.save()
        return f"Ошибка инициализации API хранилища {cloud_storage.name}: {e}, обработка прервана"


    if files:
        #Выборочная синхронизация Облачного хранилища по списку id существующих сетевых документов
        synchronization_type = "bulk"
        storage_update_report.content["type"] = "bulk"
        db_documents = NetworkDocument.objects.filter(storage=cloud_storage, pk__in=files)
        storage_files = None # TODO
    else:
        # Полная синхронизация Облачного хранилища с сетевым диском
        synchronization_type = "all"
        storage_update_report.content["type"] = "all"
        db_documents = NetworkDocument.objects.filter(storage=cloud_storage)
        storage_files = cloud_storage_api.list_directory(path=cloud_storage_api.root_path)


    total_counter = len(storage_files)
    if total_counter == 0:
        return "Обработка завершена"

    progress_recorder = ProgressRecorder(self)
    progress_now, current = 0, 0
    progress_step = ceil(total_counter / 100)
    progress_description = f'Обрабатывается {total_counter} объектов'

    result = {
        'new_files': [],
        'updated_files': [],
        'deleted_files': [],
        'restored_files': [],
        'excluded_files': [],
        'error': None
    }

    db_documents_by_url = {doc.url: doc for doc in db_documents}
    db_urls_set = set(db_documents_by_url.keys())

    incoming_urls_set = set(file["url"] for file in storage_files)

    # DELETED: те, что были в базе, но отсутствуют в облаке
    if synchronization_type == "all": #TODO надо обдумать вариант bulk
        deleted_urls_set = db_urls_set - incoming_urls_set
        deleted_docs = db_documents.filter(url__in=deleted_urls_set)
        for doc in deleted_docs:
            result["deleted_files"].append(
                {
                    "url": doc.url,
                    "name": doc.name,
                    "status": doc.status,
                    # можно добавить другие поля по необходимости
                }
            )

    # NEW / UPDATED / RESTORED / EXCLUDED
    for index, file in enumerate(storage_files):
        url = file.get('url')
        doc = db_documents_by_url.get(url)

        if url not in db_urls_set:
            result['new_files'].append(file)
        else:
            if doc.status == ContentStatus.DELETED.value:
                result['restored_files'].append(file)
            elif doc.status == ContentStatus.EXCLUDED.value:
                result['excluded_files'].append(file)
            else:
                result['updated_files'].append(file)

        if current == (progress_now + 1) * progress_step:
            progress_now += 1
            progress_recorder.set_progress(progress_now, 100, description=progress_description)
    # print(result)

    storage_update_report.content["result"] = result
    storage_update_report.save(update_fields=["content"])

    if result.get("new_files"):
        created_ids = []
        bulk_container = []
        # Формируем объекты для массового создания, пропуская дубликаты
        for file_data in result.get("new_files"):
            bulk_container.append(NetworkDocument(
                storage=cloud_storage,
                report=storage_update_report,
                title=file_data["file_name"],
                path=file_data["path"],
                file_id=file_data["file_id"],
                url=file_data["url"],
            ))
            if len(bulk_container) >= 500:
                created_docs = NetworkDocument.objects.bulk_create(bulk_container)
                created_ids.extend([doc.id for doc in created_docs])
        if bulk_container:
            created_docs = NetworkDocument.objects.bulk_create(bulk_container)
            created_ids.extend([doc.id for doc in created_docs])
        if created_ids:
            # Запускаем фоновую задачу для скачивания и создания raw content
            task = download_and_create_raw_content_parallel.delay(
                document_ids=created_ids,
                update_report_pk=update_report_pk,
            )
            storage_update_report.running_background_tasks[task.id] = "Загрузка контента новых файлов с облачного хранилища"
            storage_update_report.save(update_fields=["running_background_tasks"])

    return "Обработка завершена"


def download_and_process_file(
        doc: NetworkDocument,
        cloud_storage: CloudStorage,
        storage_update_report: CloudStorageUpdateReport,
        author: User):
    """
    Загружает файл из облачного хранилища и сохраняет его как RawContent в базу данных.
        :param author: (User): пользователь, связанный с загруженным контентом.
        :param cloud_storage: (CloudStorage): объект облачного хранилища с методом get_storage()
        :param doc: (NetworkDocument): документ, содержащий информацию об исходном файле.
        :param storage_update_report: (CloudStorageUpdateReport): отчет в рамках которого выполняется обновление
    """
    temp_path = None
    file_name = None

    logger.info(f"[Document {doc.pk}] Начинаем обработку. URL: {doc.url}")

    try:
        # Получаем объект API облачного хранилища (например, S3, WebDAV и т.д.)
        cloud_storage_api = cloud_storage.get_storage()

        # Скачиваем файл во временное хранилище на диск
        temp_path, file_name = cloud_storage_api.download_file_to_disk_sync(doc.url)
        logger.info(f"[Document {doc.pk}] Файл успешно загружен во временное хранилище: {temp_path}")
    except Exception as e:
        doc.error_message = f"Ошибка при загрузке файла: {e}"
        doc.save()
        logger.error(f"[Document {doc.pk}] Ошибка при загрузке: {e}")
        return doc.pk, "fail"

    try:
        # Создаём объект RawContent, связанный с этим документом
        raw_content = RawContent.objects.create(
            network_document=doc,
            report=storage_update_report,
            author=author)

        # Сохраняем файл в поле FileField модели RawContent
        with open(temp_path, 'rb') as f:
            raw_content.file.save(file_name, File(f), save=False)

        # Вычисляем и сохраняем хеш
        raw_content.hash_content = compute_sha512(temp_path)
        raw_content.save()
        doc.save()

        logger.info(f"[Document {doc.pk}] Файл успешно обработан и сохранён.")
    except Exception as e:
        doc.error_message = f"Ошибка при сохранении файла в БД: {e}"
        doc.save()
        logger.error(f"[Document {doc.pk}] Ошибка при сохранении в БД: {e}")
        return doc.pk, "fail"
    finally:
        # Удаляем временный файл после обработки
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug(f"[Document {doc.pk}] Временный файл удалён: {temp_path}")

    return doc.pk, "success"


@shared_task(bind=True)
def download_and_create_raw_content_parallel(self,
                                             document_ids: list[int],
                                             update_report_pk: int,
                                             max_workers: int = 5):
    """
    Задача для параллельной загрузки и обработки файлов из облачного хранилища.

    Args:
        self: ссылка на текущую задачу Celery.
        document_ids (list[int]): список ID документов для обработки.
        update_report_id (int): ID отчета обновления облака.
        author (User): пользователь, запустивший задачу.
        max_workers (int): максимальное количество потоков для обработки.

    Returns:
        str: сообщение об окончании задачи.
    """
    storage_update_report = CloudStorageUpdateReport.objects.select_related("storage", "author").get(pk=update_report_pk)
    cloud_storage = storage_update_report.storage
    documents = NetworkDocument.objects.filter(pk__in=document_ids)
    author = storage_update_report.author

    total_counter = len(documents)
    if total_counter == 0:
        return "Обработка завершена: документы не найдены"

    # Инициализация прогресса
    progress_recorder = ProgressRecorder(self)
    progress_description = f'Обрабатывается {total_counter} объектов'
    progress_percent = 0

    current = 0
    results = []

    # Параллельная загрузка и обработка
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_and_process_file, doc=doc, cloud_storage=cloud_storage,
                            author=author, storage_update_report=storage_update_report)
            for doc in documents
        ]

        for future in as_completed(futures):
            current += 1
            try:
                result = future.result()
            except Exception as e:
                logger.exception(f"Ошибка обработки документа: {e}")
                result = (None, "fail")

            results.append(result)

            # Обновление прогресса по процентам
            new_percent = int((current / total_counter) * 100)
            if new_percent > progress_percent:
                progress_percent = new_percent
                progress_recorder.set_progress(progress_percent, 100, description=progress_description)

    # Разделение результатов на успешные и неудачные
    success = [pk for pk, status in results if status == "success"]
    failed = [pk for pk, status in results if status == "fail"]

    return "Обработка завершена"


@shared_task
def download_and_create_raw_content(document_ids: list[int], update_report_id: int, max_workers: int = 5):
    update_report = CloudStorageUpdateReport.objects.select_related("storage").get(pk=update_report_id)
    cloud_storage = update_report.storage
    documents = NetworkDocument.objects.filter(pk__in=document_ids)

    results = []
    for doc in documents:
        download_and_process_file(doc=doc, cloud_storage=cloud_storage)

    success = [pk for pk, status in results if status == "success"]
    failed = [pk for pk, status in results if status == "fail"]

    logger.info(f"Обработка завершена. Успешно: {len(success)}, Ошибки: {len(failed)}")

