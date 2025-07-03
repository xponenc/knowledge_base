import asyncio
import logging
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from math import ceil
from pprint import pprint
from typing import List, Tuple

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.core.files import File
from django.core.files.base import ContentFile
from django.db.models import Subquery, OuterRef, Prefetch
from django.shortcuts import get_object_or_404
from django.utils import timezone
from dateutil.parser import parse

from app_sources.content_models import ContentStatus, RawContent, CleanedContent
from app_sources.report_models import CloudStorageUpdateReport, ReportStatus
from app_sources.services.content_process import summarize_and_save_to_document, recognize_raw_content, \
    recognize_and_summarize_content
from app_sources.services.summary import summarize_with_sber, summarize_text
from app_sources.source_models import NetworkDocument, SourceStatus
from app_sources.storage_models import CloudStorage
from app_tasks.models import TaskForSource, ContentComparison
from recognizers.dispatcher import ContentRecognizerDispatcher
from utils.process_files import compute_sha512
from django.contrib.auth import get_user_model

User = get_user_model()

logger = logging.getLogger(__name__)

# старая от 03/07
# def get_documents_for_redownload(
#         documents: List[NetworkDocument],
#         cloud_storage: CloudStorage,
#         progress_recorder: ProgressRecorder = None,
#         total: int = None,
# ) -> List[Tuple[int, str | None, str | None]]:
#     """
#     Проверяет, какие документы нужно перезагрузить из облачного хранилища.
#
#     Для каждого документа скачивается временный файл, вычисляется SHA512-хеш.
#     Если хеш отличается от сохранённого, документ попадает в результат.
#     При ошибках также документ включается для перезагрузки.
#
#     :param documents: список документов NetworkDocument с аннотацией last_hash
#     :param cloud_storage: объект облачного хранилища с get_storage()
#     :param progress_recorder: (optional) объект ProgressRecorder из Celery
#     :param total: (optional) общее число документов для прогресса
#     :return: список (doc.id, temp_path или None, file_name или None)
#     """
#     docs_to_download = []
#     total = total or len(documents)
#
#     for i, doc in enumerate(documents, 1):
#         try:
#             temp_path, file_name = cloud_storage.get_storage().download_file_to_disk_sync(doc.url)
#             current_hash = compute_sha512(temp_path)
#
#             if doc.last_hash != current_hash:
#                 logger.info(f"NetworkDocument [id {doc.pk}] Хеш изменился, файл будет перезаписан")
#                 docs_to_download.append((doc.id, temp_path, file_name))
#             else:
#                 logger.info(f"NetworkDocument [id {doc.pk}] Хеш совпадает, файл пропускается")
#                 if os.path.exists(temp_path):
#                     os.remove(temp_path)
#         except Exception as e:
#             logger.warning(f"NetworkDocument [id {doc.pk}] Ошибка при проверке хеша: {e}")
#             docs_to_download.append((doc.id, None, None))
#
#         if progress_recorder and i % 10 == 0:
#             progress_recorder.set_progress(i, total, description="Проверка документов на изменение")
#
#     return docs_to_download


def get_documents_for_redownload(
        documents: List[NetworkDocument],
        cloud_storage: CloudStorage,
        progress_recorder: ProgressRecorder = None,
        total: int = None,
) -> List[Tuple[int, str | None, str | None]]:
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
                temp_path = None  # передаём владельцу списка, не удаляем здесь
            else:
                logger.info(f"NetworkDocument [id {doc.pk}] Хеш совпадает, файл пропускается")

        except Exception as e:
            logger.warning(f"NetworkDocument [id {doc.pk}] Ошибка при проверке хеша: {e}")
            docs_to_download.append((doc.id, None, None))

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {temp_path}: {e}")

        if progress_recorder and i % 10 == 0:
            progress_recorder.set_progress(i, total, description="Проверка документов на изменение")

    return docs_to_download



def create_task_for_network_document(doc: NetworkDocument,
                                     storage_update_report: CloudStorageUpdateReport,
                                     current_raw_content: RawContent,
                                     new_raw_content: RawContent):
    """Создание задачи на изменения в системе при изменении контента сетевого документа NetworkDocument"""
    content_comparison = ContentComparison.objects.create(
        content_type="raw_content",
        old_raw_content=current_raw_content,
        new_raw_content=new_raw_content,
    )

    task = TaskForSource(
        cloud_report=storage_update_report,
        network_document=doc,
        comparison=content_comparison,
        source_previous_status=doc.status,

    )
    if doc.status == SourceStatus.READY.value:
        task.description = ("Обновился исходный контент для АКТИВНОГО источника,"
                            " необходимо подтвердить принятие нового контента")
    elif doc.status == SourceStatus.DELETED.value:
        task.description = ("Обновился исходный контент для УДАЛЕННОГО источника,"
                            " необходимо проанализировать новый контента")
    elif doc.status == SourceStatus.EXCLUDED.value:
        task.description = ("Обновился исходный контент для ИСКЛЮЧЕННОГО источника,"
                            " необходимо проанализировать новый контента")
    elif doc.status == SourceStatus.ERROR.value:
        task.description = ("Обновился исходный контент для источника в статусе ОШИБКА,"
                            " необходимо проанализировать новый контента")
    elif doc.status == SourceStatus.WAIT.value:
        task.description = ("Обновился исходный контент для источника в статусе ОБРАБОТКА,"
                            "возможен конфликт задач, необходимо проанализировать новый контента")

    if task.description:
        task.save()
    else:
        logger.error(f"При обработке NetworkDocument[id {doc.pk}] получен неизвестный статус документа")


# @shared_task(bind=True)
# def process_cloud_files(
#         self,
#         files: list[dict],
#         update_report_pk: int,
#         recognize_content: bool = False,
#         do_summarization: bool =False
# ):
#     """
#     Синхронизирует файлы из облака с локальной базой, обновляет документы и их содержимое.
#
#     Основные этапы:
#     1. Инициализация API облачного хранилища.
#     2. Определение типа синхронизации:
#        - 'bulk' — выборочная синхронизация по списку файлов (передаются ID документов).
#        - 'all' — полная синхронизация всего хранилища.
#     3. Получение текущих документов из базы и списка файлов из облака.
#     4. Категоризация файлов по статусу: новые, существующие, удалённые и т.п.
#     5. Создание новых записей NetworkDocument для новых файлов.
#     6. Проверка существующих документов на необходимость обновления контента.
#        Сравнение хешей для выявления изменённых файлов.
#     7. Создание новых RawContent только для изменённых файлов, пропуск неизменённых.
#     8. Обновление отчёта о синхронизации.
#
#     :param self: Контекст задачи Celery.
#     :param files: Список файлов для выборочной синхронизации по id документов (bulk).
#         Если пустой список, происходит полная синхронизация.
#     :param update_report_pk: ID объекта отчёта CloudStorageUpdateReport для логирования.
#     :param recognize_content: Выполнить попытку распознавания файла и создание CleanedContent
#     :param do_summarization: При включенном recognize_content выполнить попытку саммаризации файла
#      и записать резальтат в description CleanedContent
#     :return: Строка с результатом обработки.
#     :raises: None
#     """
#     # Получаем объект отчёта обновления для записи результатов и доступа к storage
#     storage_update_report = CloudStorageUpdateReport.objects.select_related("storage").get(pk=update_report_pk)
#     cloud_storage = storage_update_report.storage
#
#     # Инициализация API облачного хранилища (S3, WebDAV и т.п.)
#     try:
#         cloud_storage_api = cloud_storage.get_storage()
#         logger.info(f"API хранилища {cloud_storage.name} успешно инициализировано")
#     except ValueError as e:
#         # Если не удалось инициализировать API, записываем ошибку в отчёт и завершаем задачу
#         logger.error(f"Ошибка инициализации API хранилища {cloud_storage.name}: {e}")
#         storage_update_report.content.setdefault("errors", []).append(str(e))
#         storage_update_report.status = ReportStatus.ERROR.value
#         storage_update_report.save()
#         return f"Ошибка инициализации API хранилища {cloud_storage.name}: {e}, обработка прервана"
#
#     # Определяем режим синхронизации: bulk — выборочная по списку, all — полная
#     if files:
#         synchronization_type = "bulk"
#         storage_update_report.content["type"] = "bulk"
#         # Получаем документы из базы по списку ID
#         db_documents = NetworkDocument.objects.filter(storage=cloud_storage, pk__in=files)
#         storage_files = None  # TODO: реализовать получение метаинформации по bulk
#     else:
#         synchronization_type = "all"
#         storage_update_report.content["type"] = "all"
#         # Получаем все документы из базы и полный список файлов из облака
#         db_documents = NetworkDocument.objects.filter(storage=cloud_storage)
#         storage_files = cloud_storage_api.list_directory(path=cloud_storage_api.root_path)
#
#     total_counter = len(storage_files) if storage_files else 0
#     if total_counter == 0:
#         # Если файлов нет, завершаем работу
#         return "Обработка завершена"
#
#     # Инициализируем прогресс-бар Celery
#     progress_recorder = ProgressRecorder(self)
#     progress_now, current = 0, 0
#     progress_step = max(1, ceil(total_counter / 100))
#     progress_description = f'Обрабатывается {total_counter} объектов'
#
#     # Подготовка структуры результата синхронизации
#     result = {
#         'new_files': [],
#         'exist_documents': [],
#         'skipped_documents': [],
#         'updated_documents': [],
#         'error': None
#     }
#
#     # Словарь для быстрого доступа к документам по url
#     db_documents_by_url = {doc.url: doc for doc in db_documents}
#     incoming_urls_set = set(file["url"] for file in storage_files)
#
#     # Определяем документы, отсутствующие в облаке, считаем их удалёнными
#     if synchronization_type == "all":
#         deleted_urls_set = set(db_documents_by_url) - incoming_urls_set
#         deleted_docs = db_documents.filter(url__in=deleted_urls_set)
#         for doc in deleted_docs:
#             result.setdefault("deleted_files", []).append({
#                 "url": doc.url,
#                 "name": doc.title if hasattr(doc, "title") else "",
#                 "status": doc.status,
#             })
#
#     # Классифицируем файлы: новые и существующие
#     for index, file in enumerate(storage_files):
#         url = file.get('url')
#         if url not in db_documents_by_url:
#             result['new_files'].append(file)
#         else:
#             # result['exist_documents'].append(db_documents_by_url[url])
#             result['exist_documents'].append(db_documents_by_url[url].pk)
#
#         # Обновляем прогресс
#         if current == (progress_now + 1) * progress_step:
#             progress_now += 1
#             progress_recorder.set_progress(progress_now, 100, description=progress_description)
#
#     # Сохраняем промежуточный результат в отчёте
#     storage_update_report.content["result"] = result
#     storage_update_report.save(update_fields=["content"])
#
#     created_ids = []
#     # Обрабатываем новые файлы — создаём документы
#     if result.get("new_files"):
#         bulk_container = []
#         for file_data in result.get("new_files"):
#             bulk_container.append(NetworkDocument(
#                 storage=cloud_storage,
#                 report=storage_update_report,
#                 title=file_data["file_name"],
#                 path=file_data["path"],
#                 file_id=file_data["file_id"],
#                 url=file_data["url"],
#             ))
#             if len(bulk_container) >= 500:
#                 created_docs = NetworkDocument.objects.bulk_create(bulk_container)
#                 created_ids.extend([doc.id for doc in created_docs])
#                 bulk_container.clear()
#         if bulk_container:
#             created_docs = NetworkDocument.objects.bulk_create(bulk_container)
#             created_ids.extend([doc.id for doc in created_docs])
#
#         # Запускаем фоновую задачу для загрузки контента новых файлов
#         if created_ids:
#             task = download_and_create_raw_content_parallel.delay(
#                 document_ids=created_ids,
#                 update_report_pk=update_report_pk,
#             )
#             storage_update_report.running_background_tasks[
#                 task.id] = "Загрузка контента новых файлов с облачного хранилища"
#             storage_update_report.save(update_fields=["running_background_tasks"])
#
#
#     # Обрабатываем существующие документы, проверяя их хеши для обновления RawContent
#     if result.get("exist_documents"):
#         # Выбираем документы для проверки обновлений
#         # doc_ids_to_check = [
#         #     db_documents_by_url[file["url"]].id
#         #     for file in result.get("exist_documents")
#         #     if file["url"] in db_documents_by_url
#         # ]
#         doc_ids_to_check = result.get("exist_documents")
#
#         # Берется последний RawContent со статусом READY для проверки изменений контента
#         docs_to_check = NetworkDocument.objects.filter(pk__in=doc_ids_to_check).annotate(
#             last_hash=Subquery(
#                 RawContent.objects.filter(
#                     network_document=OuterRef("pk"),
#                     status=ContentStatus.READY.value,
#                 ).order_by("-created_at").values("hash_content")[:1]
#             )
#         )
#
#         # Обновляем прогресс до вызова get_documents_for_redownload
#         progress_recorder = ProgressRecorder(self)
#
#         # Проверяем какие документы требуют перезагрузки контента
#         docs_to_download = get_documents_for_redownload(
#             documents=docs_to_check,
#             cloud_storage=cloud_storage,
#             progress_recorder=progress_recorder,
#         )
#
#         # Разбиваем на документы с готовыми временными файлами и без
#         updated_docs_ids = [item[0] for item in docs_to_download]
#         updated_docs_with_files = [item for item in docs_to_download if item[1] is not None]
#         updated_docs_without_files = [item for item in docs_to_download if item[1] is None]
#
#         # Выделяем id документов, контент которых не изменился — их пропускаем
#         skipped_doc_ids = [doc.pk for doc in docs_to_check if
#                            doc.pk not in [doc_id for doc_id, _, _ in docs_to_download]]
#         result['skipped_documents'].extend(skipped_doc_ids)
#
#         updated_doc_ids = []
#
#         raw_content_qs = RawContent.objects.filter(
#             status=ContentStatus.READY.value
#         ).order_by("-created_at")[:1]
#
#         # raw_content_qs = Prefetch(
#         #     'rawcontent_set',
#         #     queryset=RawContent.objects.filter(status=ContentStatus.READY.value).order_by('-created_at'),
#         #     to_attr='related_rawcontents'
#         # )
#
#         docs_to_update = NetworkDocument.objects.filter(pk__in=updated_docs_ids).prefetch_related(
#             Prefetch('rawcontent_set', queryset=raw_content_qs, to_attr='related_rawcontents'))
#
#         for doc in docs_to_update:
#             doc.current_raw_content = doc.related_rawcontents[0] if doc.related_rawcontents else None
#
#         docs_to_update_data = {doc.id: doc for doc in docs_to_update}
#
#         # Для документов с временными файлами — сразу создаём RawContent
#         for doc_id, temp_path, file_name in updated_docs_with_files:
#             try:
#                 doc = docs_to_update_data.get(doc_id)
#                 current_raw_content = doc.current_raw_content
#                 new_raw_content = RawContent(
#                     network_document=doc,
#                     report=storage_update_report,
#                     author=storage_update_report.author,
#                 )
#
#                 with open(temp_path, "rb") as f:
#                     new_raw_content.file.save(file_name, File(f), save=False)
#                 new_raw_content.hash_content = compute_sha512(temp_path)
#                 new_raw_content.save()
#                 updated_doc_ids.append(doc_id)
#                 logger.info(f"[Document {doc.pk}] RawContent обновлён из кэша temp_path")
#
#                 # Создание задачи для действий по изменению
#                 print(f"{doc.current_raw_content=}")
#                 print(f"{new_raw_content=}")
#                 try:
#                     create_task_for_network_document(doc=doc,
#                                                      storage_update_report=storage_update_report,
#                                                      current_raw_content=current_raw_content,
#                                                      new_raw_content=new_raw_content)
#                 except Exception as e:
#                     logger.error(f"Ошибка при создании задачи на изменение контента NetworkDocument [id {doc.pk}]: {e}")
#                     new_raw_content.status = ContentStatus.ERROR.value
#                     new_raw_content.error_message = (f"Ошибка при создании задачи на изменение контента"
#                                                  f" NetworkDocument [id {doc.pk}]: {e}")
#                     new_raw_content.save()
#
#
#             except Exception as e:
#                 storage_update_report.content.setdefault("errors", []).append(
#                     f"[Document {doc_id}] Ошибка при создании RawContent из temp_path: {e}")
#                 storage_update_report.status = ReportStatus.ERROR.value
#                 logger.error(f"[Document {doc_id}] Ошибка при создании RawContent из temp_path: {e}")
#             finally:
#                 if temp_path and os.path.exists(temp_path):
#                     os.remove(temp_path)
#
#         # Для документов без временных файлов — ставим задачу на загрузку
#         if updated_docs_without_files:
#             document_ids_to_dl = [doc_id for doc_id, _, _ in updated_docs_without_files]
#             task = download_and_create_raw_content_parallel.delay(
#                 document_ids=document_ids_to_dl,
#                 update_report_pk=update_report_pk,
#             )
#             storage_update_report.running_background_tasks[task.id] = "Загрузка контента для обновления файлов"
#             storage_update_report.save(update_fields=["running_background_tasks"])
#
#         result['updated_documents'].extend(updated_doc_ids)
#
#     # Обновляем отчёт с итогами обработки
#     storage_update_report.content["result"] = result
#     storage_update_report.save(update_fields=["content"])
#
#     logger.error(f"{recognize_content=}")
#     # if recognize_content:
#     # Получаем все актуальные raw_content
#     processed_doc_ids = result.get("updated_documents", []) + created_ids
#     raw_contents = RawContent.objects.filter(
#         network_document__in=processed_doc_ids,
#         status=ContentStatus.READY.value
#     ).select_related("network_document")
#     logger.error(raw_contents)
#     total = raw_contents.count()
#     if total > 0:
#         progress_description = f"Распознавание и саммаризация ({total} файлов)"
#         progress_step = max(1, ceil(total / 100))
#         progress_recorder.set_progress(0, 100, description=progress_description)
#
#         for i, raw_content in enumerate(raw_contents):
#             try:
#                 recognize_and_summarize_content(raw_content=raw_content,
#                                                 user_id=storage_update_report.author_id,
#                                                 do_summarization=do_summarization
#                                                 )
#             except Exception as e:
#                 logger.warning(f"Ошибка при распознавании RawContent {raw_content.id}: {e}")
#                 raw_content.status = ContentStatus.ERROR.value
#                 raw_content.error_message = str(e)
#                 raw_content.save()
#
#             if (i + 1) % progress_step == 0:
#                 progress_recorder.set_progress(i + 1, total, description=progress_description)
#
#         progress_recorder.set_progress(100, 100, description="Распознавание завершено")
#
#     return "Обработка завершена"


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
        storage_update_report.content.setdefault("errors", []).append(
            f"[Document {doc.pk}] Ошибка при загрузке: {e}")
        storage_update_report.status = ReportStatus.ERROR.value

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

        logger.info(f"[Document {doc.pk}] Файл успешно обработан и сохранён.")
        if doc.status != SourceStatus.CREATED.value:
            # Создание задачи для действий по изменению
            try:
                create_task_for_network_document(doc=doc,
                                                 storage_update_report=storage_update_report,
                                                 current_raw_content=doc.current_raw_content,
                                                 new_raw_content=raw_content)
            except Exception as e:
                logger.error(f"Ошибка при создании задачи на изменение контента NetworkDocument [id {doc.pk}]: {e}")
                raw_content.status = ContentStatus.ERROR.value
                raw_content.error_message = (f"Ошибка при создании задачи на изменение контента"
                                             f" NetworkDocument [id {doc.pk}]: {e}")
                raw_content.save()
        else:
            doc.status = SourceStatus.READY.value
            doc.save(update_fields=["status", ])
    except Exception as e:
        storage_update_report.content.setdefault("errors", []).append(
            f"[Document {doc.pk}] Ошибка при сохранении файла в БД: {e}")
        storage_update_report.status = ReportStatus.ERROR.value
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
        :param self: ссылка на текущую задачу Celery.
        :param max_workers:  максимальное количество потоков для обработки
        :param document_ids: список ID документов NetworkDocument для обработки
        :param update_report_pk: ID отчета обновления облака.
    """
    storage_update_report = CloudStorageUpdateReport.objects.select_related("storage", "author").get(
        pk=update_report_pk)
    cloud_storage = storage_update_report.storage

    raw_content_qs = RawContent.objects.filter(
        status=ContentStatus.READY.value
    ).order_by("-created_at")[:1]

    documents = NetworkDocument.objects.filter(pk__in=document_ids).prefetch_related(
        Prefetch('rawcontent_set', queryset=raw_content_qs, to_attr='related_rawcontents'))

    for doc in documents:
        doc.current_raw_content = doc.related_rawcontents[0] if doc.related_rawcontents else None

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


# @shared_task
# def download_and_create_raw_content(document_ids: list[int], update_report_id: int, max_workers: int = 5):
#     update_report = CloudStorageUpdateReport.objects.select_related("storage").get(pk=update_report_id)
#     cloud_storage = update_report.storage
#     documents = NetworkDocument.objects.filter(pk__in=document_ids)
#
#     results = []
#     for doc in documents:
#         download_and_process_file(doc=doc, cloud_storage=cloud_storage)
#
#     success = [pk for pk, status in results if status == "success"]
#     failed = [pk for pk, status in results if status == "fail"]
#
#     logger.info(f"Обработка завершена. Успешно: {len(success)}, Ошибки: {len(failed)}")


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

#
# @shared_task(bind=True)
# def process_raw_content_batch_task(self, raw_content_ids: list[int], user_id:int, report_id:int):
#     """
#     Пакетная обработка списка RawContent: распознавание и саммаризация.
#     """
#     progress_recorder = ProgressRecorder(self)
#
#     total = len(raw_content_ids)
#     for i, raw_content_id in enumerate(raw_content_ids, start=1):
#         progress_recorder.set_progress(i - 1, total, description=f"Обрабатывается {raw_content_id}")
#
#         try:
#             raw_content = RawContent.objects.get(pk=raw_content_id)
#
#             # 1. Распознавание
#             recognition = recognize_raw_content(raw_content)
#             recognized_text = recognition["text"]
#
#             # 2. Очистка старого и создание нового CleanedContent
#             CleanedContent.objects.filter(raw_content=raw_content).delete()
#             cleaned = CleanedContent.objects.create(
#                 network_document=raw_content.network_document,
#                 report_id=report_id,
#                 raw_content=raw_content,
#                 recognition_method=recognition["method"],
#                 recognition_quality=recognition["quality_report"],
#                 preview=recognized_text[:200],
#                 author_id=user_id
#             )
#             cleaned.file.save("cleaned.txt", ContentFile(recognized_text.encode("utf-8")))
#             cleaned.save()
#
#             # 3. Саммаризация
#             summarize_and_save_to_document(recognized_text, raw_content.network_document)
#
#         except Exception as e:
#             # Можно логировать ошибку в базу или отдельный лог
#             print(f"Ошибка при обработке raw_content_id={raw_content_id}: {e}")
#             continue
#
#         progress_recorder.set_progress(i, total, description=f"Готово: {raw_content_id}")
#
#     return "Готово"


from typing import List, Dict, Tuple, Optional
from celery_progress.backend import ProgressRecorder
import logging
import os

document_logger = logging.getLogger('document_processing')
logger = logging.getLogger(__name__)

@shared_task(bind=True)
def process_cloud_files(
    self,
    files: List[Dict],
    update_report_pk: int,
    recognize_content: bool = False,
    do_summarization: bool = False
):
    """
    Основная задача для обработки файлов в облачном хранилище.

    Args:
        self: Ссылка на задачу Celery (для прогресса).
        files: Список словарей с информацией о файлах для выборочной обработки.
        update_report_pk: PK отчёта обновления в БД.
        recognize_content: Флаг запуска распознавания текста и суммаризации.
        do_summarization: Флаг запуска суммаризации.

    Returns:
        Строку с результатом выполнения.
    """

    # Получаем объект отчёта обновления
    storage_update_report = get_update_report(update_report_pk)

    # Инициализируем API облачного хранилища
    cloud_storage_api = initialize_cloud_storage_api(storage_update_report)

    # Получаем тип синхронизации, документы из базы и файлы из хранилища
    synchronization_type, db_documents, storage_files = fetch_storage_files(files, cloud_storage_api, storage_update_report)

    # Классифицируем файлы: новые, существующие, удалённые
    result = classify_documents(
        synchronization_type,
        db_documents,
        storage_files,
        storage_update_report,
        self
    )

    # Создаём новые документы в БД для новых файлов
    created_ids = create_new_documents(result.get("new_files", []), storage_update_report, self)

    # Обрабатываем существующие документы, обновляя контент при необходимости
    updated_doc_ids = process_existing_documents(result.get("exist_documents", []), storage_update_report, self)

    # Выполняем распознавание и суммаризацию (если нужно)
    final_recognition_and_summary(
        storage_update_report,
        recognize_content,
        do_summarization,
        self,
        created_ids + updated_doc_ids
    )

    # Обновляем отчёт с итогами обработки
    update_report_with_results(storage_update_report, result)

    return "Обработка завершена"


def get_update_report(update_report_pk: int):
    """
    Получает объект отчёта обновления по PK.

    Args:
        update_report_pk: PK отчёта обновления.

    Returns:
        Объект CloudStorageUpdateReport.
    """
    return CloudStorageUpdateReport.objects.select_related("storage").get(pk=update_report_pk)


def initialize_cloud_storage_api(storage_update_report):
    """
    Инициализирует API облачного хранилища из объекта отчёта.

    Args:
        storage_update_report: Объект отчёта обновления.

    Returns:
        Инициализированный API облачного хранилища.

    Raises:
        ValueError при ошибке инициализации.
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
    Получает файлы из облачного хранилища и документы из базы.

    Если передан список files — работает в режиме bulk (частичная выборка),
    иначе — получает все файлы из хранилища.

    Args:
        files: Список файлов для выборочной обработки.
        cloud_storage_api: API облачного хранилища.
        storage_update_report: Объект отчёта обновления.

    Returns:
        Кортеж: (тип синхронизации, QuerySet документов из БД, список файлов из хранилища).
    """
    cloud_storage = storage_update_report.storage
    if files:
        synchronization_type = "bulk"
        storage_update_report.content["type"] = "bulk"
        db_documents = NetworkDocument.objects.filter(storage=cloud_storage, pk__in=files)
        storage_files = None  # TODO: реализовать получение метаинформации для bulk
    else:
        synchronization_type = "all"
        storage_update_report.content["type"] = "all"
        db_documents = NetworkDocument.objects.filter(storage=cloud_storage)
        storage_files = cloud_storage_api.list_directory(path=cloud_storage_api.root_path)
    return synchronization_type, db_documents, storage_files


def classify_documents(synchronization_type, db_documents, storage_files, storage_update_report, task_self):
    """
    Классифицирует файлы на новые, существующие и удалённые, с обновлением прогресса.

    Args:
        synchronization_type: Тип синхронизации ('bulk' или 'all').
        db_documents: QuerySet документов из БД.
        storage_files: Список файлов из хранилища.
        storage_update_report: Объект отчёта обновления.
        task_self: Ссылка на задачу Celery для прогресса.

    Returns:
        Словарь с классификацией файлов (new_files, exist_documents и пр.).
    """
    total_counter = len(storage_files) if storage_files else 0
    if total_counter == 0:
        return {}

    progress_recorder = ProgressRecorder(task_self)
    progress_now, current = 0, 0
    progress_step = max(1, (total_counter // 100))
    progress_description = f'Обрабатывается {total_counter} объектов'

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

    # Для полной синхронизации вычисляем удалённые документы
    if synchronization_type == "all":
        deleted_urls_set = set(db_documents_by_url) - incoming_urls_set
        deleted_docs = db_documents.filter(url__in=deleted_urls_set)
        for doc in deleted_docs:
            result["deleted_files"].append({
                "url": doc.url,
                "name": getattr(doc, "title", ""),
                "status": doc.status,
            })

    # Классифицируем каждый файл как новый или существующий
    for index, file in enumerate(storage_files or []):
        url = file.get('url')
        if url not in db_documents_by_url:
            result['new_files'].append(file)
        else:
            result['exist_documents'].append(db_documents_by_url[url].pk)

        # Обновляем прогресс каждые progress_step
        if index >= (progress_now + 1) * progress_step:
            progress_now += 1
            progress_recorder.set_progress(progress_now, 100, description=progress_description)

    # Сохраняем результат классификации в отчёт
    storage_update_report.content["result"] = result
    storage_update_report.save(update_fields=["content"])
    return result


def create_new_documents(new_files, storage_update_report, task_self) -> List[int]:
    """
    Создаёт новые записи документов в базе для новых файлов.

    Args:
        new_files: Список данных новых файлов.
        storage_update_report: Объект отчёта обновления.
        task_self: Задача Celery для прогресса.

    Returns:
        Список ID созданных документов.
    """
    created_ids = []
    if not new_files:
        return created_ids

    cloud_storage = storage_update_report.storage
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
        if len(bulk_container) >= 500:
            created_docs = NetworkDocument.objects.bulk_create(bulk_container)
            created_ids.extend([doc.id for doc in created_docs])
            bulk_container.clear()

    if bulk_container:
        created_docs = NetworkDocument.objects.bulk_create(bulk_container)
        created_ids.extend([doc.id for doc in created_docs])

    if created_ids:
        # Запускаем задачу фоновой загрузки контента для новых документов
        task = download_and_create_raw_content_parallel.delay(
            document_ids=created_ids,
            update_report_pk=storage_update_report.pk,
        )
        storage_update_report.running_background_tasks[task.id] = "Загрузка контента новых файлов с облачного хранилища"
        storage_update_report.save(update_fields=["running_background_tasks"])

    return created_ids


def process_existing_documents(exist_documents, storage_update_report, task_self) -> List[int]:
    """
    Обрабатывает существующие документы, проверяя необходимость обновления.

    Args:
        exist_documents: Список PK существующих документов.
        storage_update_report: Объект отчёта обновления.
        task_self: Задача Celery для прогресса.

    Returns:
        Список ID документов, для которых выполнено обновление контента.
    """
    updated_doc_ids = []
    if not exist_documents:
        return updated_doc_ids

    cloud_storage = storage_update_report.storage

    # Получаем документы с аннотированным хешем последнего RawContent
    docs_to_check = NetworkDocument.objects.filter(pk__in=exist_documents).annotate(
        last_hash=Subquery(
            RawContent.objects.filter(
                network_document=OuterRef("pk"),
                status=ContentStatus.READY.value,
            ).order_by("-created_at").values("hash_content")[:1]
        )
    )

    progress_recorder = ProgressRecorder(task_self)

    # Получаем документы, для которых нужно перезагрузить контент
    docs_to_download = get_documents_for_redownload(
        documents=docs_to_check,
        cloud_storage=cloud_storage,
        progress_recorder=progress_recorder,
    )

    updated_docs_ids = [item[0] for item in docs_to_download]
    updated_docs_with_files = [item for item in docs_to_download if item[1] is not None]
    updated_docs_without_files = [item for item in docs_to_download if item[1] is None]

    # Обрабатываем документы с готовыми файлами в кэше
    raw_content_qs = RawContent.objects.filter(
        status=ContentStatus.READY.value
    ).order_by("-created_at")[:1]

    docs_to_update = NetworkDocument.objects.filter(pk__in=updated_docs_ids).prefetch_related(
        Prefetch('rawcontent_set', queryset=raw_content_qs, to_attr='related_rawcontents'))

    for doc in docs_to_update:
        doc.current_raw_content = doc.related_rawcontents[0] if doc.related_rawcontents else None

    docs_to_update_data = {doc.id: doc for doc in docs_to_update}

    for doc_id, temp_path, file_name in updated_docs_with_files:
        try:
            doc = docs_to_update_data.get(doc_id)
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
            updated_doc_ids.append(doc_id)
            logger.info(f"[Document {doc.pk}] RawContent обновлён из кэша temp_path")

            # Создаём задачу на дальнейшую обработку документа
            try:
                create_task_for_network_document(
                    doc=doc,
                    storage_update_report=storage_update_report,
                    current_raw_content=current_raw_content,
                    new_raw_content=new_raw_content,
                )
            except Exception as e:
                logger.error(f"Ошибка при создании задачи на изменение контента NetworkDocument [id {doc.pk}]: {e}")
                new_raw_content.status = ContentStatus.ERROR.value
                new_raw_content.error_message = f"Ошибка при создании задачи на изменение контента NetworkDocument [id {doc.pk}]: {e}"
                new_raw_content.save()

        except Exception as e:
            storage_update_report.content.setdefault("errors", []).append(
                f"[Document {doc_id}] Ошибка при создании RawContent из temp_path: {e}")
            storage_update_report.status = ReportStatus.ERROR.value
            logger.error(f"[Document {doc_id}] Ошибка при создании RawContent из temp_path: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # Для документов без готовых файлов запускаем фоновую задачу загрузки
    if updated_docs_without_files:
        document_ids_to_dl = [doc_id for doc_id, _, _ in updated_docs_without_files]
        task = download_and_create_raw_content_parallel.delay(
            document_ids=document_ids_to_dl,
            update_report_pk=storage_update_report.pk,
        )
        storage_update_report.running_background_tasks[task.id] = "Загрузка контента для обновления файлов"
        storage_update_report.save(update_fields=["running_background_tasks"])

    return updated_doc_ids


def final_recognition_and_summary(
    storage_update_report,
    recognize_content: bool,
    do_summarization: bool,
    task_self,
    processed_doc_ids: List[int]
):
    """
    Выполняет распознавание текста и суммаризацию для обработанных документов.

    Args:
        storage_update_report: Объект отчёта обновления.
        recognize_content: Флаг включения распознавания.
        do_summarization: Флаг включения суммаризации.
        task_self: Задача Celery для прогресса.
        processed_doc_ids: Список ID обработанных документов.
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
    progress_description = f"Распознавание и суммаризация ({total} файлов)"
    progress_step = max(1, total // 100)
    progress_recorder.set_progress(0, total, description=progress_description)

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

        if (i + 1) % progress_step == 0:
            progress_recorder.set_progress(i + 1, total, description=progress_description)

    progress_recorder.set_progress(total, total, description="Распознавание завершено")


def update_report_with_results(storage_update_report, result):
    """
    Обновляет объект отчёта с результатами обработки.

    Args:
        storage_update_report: Объект отчёта обновления.
        result: Словарь с результатами классификации.
    """
    storage_update_report.content["result"] = result
    storage_update_report.save(update_fields=["content"])
