import asyncio
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from math import ceil

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.core.files import File
from django.utils import timezone

from app_sources.content_models import ContentStatus, RawContent
from app_sources.report_model import CloudStorageUpdateReport
from app_sources.source_models import NetworkDocument
from app_sources.storage_models import CloudStorage
from utils.process_files import compute_sha512
from django.contrib.auth import get_user_model

User = get_user_model()

logger = logging.getLogger(__name__)


# @shared_task
# def parse_url_task(url_id):
#     """Задача для парсинга URL."""
#     url = URL.objects.get(id=url_id)
#     result = parse_url(url)
#     if result.get('success'):
#         url.update_status('parsed')
#         chunks = generate_chunks(url)
#         for chunk in chunks:
#             embedding = generate_embedding(chunk)
#     else:
#         url.error_message = result.get('error')
#         url.update_status('error')

# @shared_task
# def sync_document_task(document_id):
#     """Задача для синхронизации документа."""
#     document = Document.objects.get(id=document_id)
#     result = sync_document(document)
#     if result.get('success'):
#         document.update_status('synced')
#         chunks = generate_chunks(document)
#         for chunk in chunks:
#             embedding = generate_embedding(chunk)
#     else:
#         document.error_message = result.get('error')
#         document.update_status('error')

@shared_task(bind=True)
def process_cloud_files(
    self,
    files: list[dict],
    cloud_storage: CloudStorage,
    update_report_pk: int
):
    """
    Синхронизирует список файлов из облачного хранилища с локальной базой данных.

    Производится предварительная категоризация файлов для последующей обработки:
    - `new_files`: файлы, которых ещё нет в БД
    - `updated_files`: файлы, уже существующие в БД и не имеющие особого статуса
    - `deleted_files`: файлы, которые были в БД, но больше не существуют в облаке
    - `restored_files`: файлы, ранее помеченные как DELETED, но снова появившиеся в облаке
    - `excluded_files`: файлы со статусом EXCLUDED, которые есть в облаке

    :param self: Контекст Celery-задачи
    :param files: Список словарей с метаинформацией по файлам из облачного хранилища
    :param cloud_storage: Объект CloudStorage, с которым идёт синхронизация
    :param update_report: Объект CloudStorageUpdateReport для хранения результата синхронизации
    :return: Строка статуса завершения
    :raises: None
    """
    total_counter = len(files)
    if total_counter == 0:
        return "Обработка завершена"

    progress_recorder = ProgressRecorder(self)
    progress_now, current = 0, 0
    progress_step = ceil(total_counter / 100)
    progress_description = f'Обрабатывается {total_counter} объектов'

    result = {
        'new_files': {},
        'updated_files': {},
        'deleted_files': {},
        'restored_files': {},
        'excluded_files': {},
        'error': None
    }

    db_documents = NetworkDocument.objects.filter(storage=cloud_storage)
    db_documents_by_url = {doc.url: doc for doc in db_documents}
    db_urls_set = set(db_documents_by_url.keys())

    incoming_urls_set = set(file["url"] for file in files)

    # DELETED: те, что были в базе, но отсутствуют в облаке
    deleted_urls_set = db_urls_set - incoming_urls_set
    deleted_docs = db_documents.filter(url__in=deleted_urls_set)
    for doc in deleted_docs:
        result["deleted_files"][doc.id] = {
            "url": doc.url,
            "name": doc.name,
            "status": doc.status,
            # можно добавить другие поля по необходимости
        }

    # NEW / UPDATED / RESTORED / EXCLUDED
    for index, file in enumerate(files):
        url = file.get('url')
        doc = db_documents_by_url.get(url)

        if url not in db_urls_set:
            result['new_files'][index] = file
        else:
            if doc.status == ContentStatus.DELETED.value:
                result['restored_files'][index] = file
            elif doc.status == ContentStatus.EXCLUDED.value:
                result['excluded_files'][index] = file
            else:
                result['updated_files'][index] = file

        if current == (progress_now + 1) * progress_step:
            progress_now += 1
            progress_recorder.set_progress(progress_now, 100, description=progress_description)
    # print(result)
    update_report = CloudStorageUpdateReport.objects.get(pk=update_report_pk)
    update_report.content["result"] = result
    update_report.content["current_status"] = "Sorting of synchronization files completed successfully"
    update_report.save(update_fields=["content"])
    return "Обработка завершена"


# def download_and_process_file(doc, cloud_storage, author):
#     temp_path = None
#     file_name = None
#     print(f"{doc.url=}")
#     try:
#         # Получаем API для скачивания
#         cloud_storage_api = cloud_storage.get_storage()
#
#         # Скачиваем файл на диск
#         temp_path, file_name = cloud_storage_api.download_file_to_disk_sync(doc.url)
#     except Exception as e:
#         doc.error_message=f"Ошибка при загрузке файла: {e}"
#         doc.status = "er"
#         doc.save()
#         logger.error(f"[Document {doc.pk}] Ошибка при загрузке: {e}")
#         return doc.pk, "fail"
#
#     try:
#         # Сохраняем файл в базу данных
#         raw_content = RawContent.objects.create(network_document=doc, author=author)
#
#         with open(temp_path, 'rb') as f:
#             raw_content.file.save(file_name, File(f), save=False)
#
#         # with open(temp_path, 'rb') as f:
#         #     content = f.read()
#         # raw_content.file.save(file_name, BytesIO(content), save=False)
#
#         raw_content.hash_content = compute_sha512(temp_path)
#         raw_content.save()
#         doc.size = raw_content.file.size
#         doc.synchronized_at = timezone.now()
#         doc.status = "sy"  # synced
#         doc.save()
#     except Exception as e:
#         doc.error_message = f"Ошибка при сохранении файла в БД: {e}"
#         doc.status = "er"  # failed to save
#         doc.save()
#         logger.error(f"[Document {doc.pk}] Ошибка при сохранении: {e}")
#         return doc.pk, "fail"
#     finally:
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)
#
#     return doc.pk, "success"


def download_and_process_file(doc, cloud_storage, author):
    """
    Загружает файл из облачного хранилища и сохраняет его как RawContent в базу данных.

    Args:
        doc (NetworkDocument): документ, содержащий информацию об исходном файле.
        cloud_storage (CloudStorage): объект облачного хранилища с методом get_storage().
        author (User): пользователь, связанный с загруженным контентом.

    Returns:
        tuple: (document_id, "success" | "fail")
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
        doc.status = "er"  # error
        doc.save()
        logger.error(f"[Document {doc.pk}] Ошибка при загрузке: {e}")
        return doc.pk, "fail"

    try:
        # Создаём объект RawContent, связанный с этим документом
        raw_content = RawContent.objects.create(network_document=doc, author=author)

        # Сохраняем файл в поле FileField модели RawContent
        with open(temp_path, 'rb') as f:
            raw_content.file.save(file_name, File(f), save=False)

        # Альтернативный способ — если нужно прочитать файл в память
        # with open(temp_path, 'rb') as f:
        #     content = f.read()
        # raw_content.file.save(file_name, BytesIO(content), save=False)

        # Вычисляем и сохраняем хеш
        raw_content.hash_content = compute_sha512(temp_path)
        raw_content.save()

        # Обновляем поля документа
        doc.size = raw_content.file.size
        doc.synchronized_at = timezone.now()
        doc.status = "sy"  # synced
        doc.save()

        logger.info(f"[Document {doc.pk}] Файл успешно обработан и сохранён.")
    except Exception as e:
        doc.error_message = f"Ошибка при сохранении файла в БД: {e}"
        doc.status = "er"
        doc.save()
        logger.error(f"[Document {doc.pk}] Ошибка при сохранении в БД: {e}")
        return doc.pk, "fail"
    finally:
        # Удаляем временный файл после обработки
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug(f"[Document {doc.pk}] Временный файл удалён: {temp_path}")

    return doc.pk, "success"

#
# @shared_task(bind=True)
# def download_and_create_raw_content_parallel(self,
#                                              document_ids: list[int],
#                                              update_report_id: int,
#                                              author: User,
#                                              max_workers: int = 5):
#     update_report = CloudStorageUpdateReport.objects.select_related("storage").get(pk=update_report_id)
#     cloud_storage = update_report.storage
#     documents = NetworkDocument.objects.filter(pk__in=document_ids)
#
#     total_counter = len(documents)
#     if total_counter == 0:
#         return "Обработка завершена"
#
#     progress_recorder = ProgressRecorder(self)
#     progress_now, current = 0, 0
#     progress_step = ceil(total_counter / 100)
#     progress_description = f'Обрабатывается {total_counter} объектов'
#
#     results = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [
#             executor.submit(download_and_process_file, doc, cloud_storage, author)
#             for doc in documents
#         ]
#         for future in as_completed(futures):
#             current += 1
#             results.append(future.result())
#
#             if current == (progress_now + 1) * progress_step:
#                 progress_now += 1
#                 progress_recorder.set_progress(progress_now, 100, description=progress_description)
#
#     success = [pk for pk, status in results if status == "success"]
#     failed = [pk for pk, status in results if status == "fail"]
#     if not failed:
#         update_report.content["current_status"] = "Content loading completed successfully"
#     else:
#         update_report.content["current_status"] = "Content loading failed"
#     update_report.save(update_fields=["content"])
#     logger.info(f"Обработка завершена. Успешно: {len(success)}, Ошибки: {len(failed)}")
#     return "Обработка завершена"


@shared_task(bind=True)
def download_and_create_raw_content_parallel(self,
                                             document_ids: list[int],
                                             update_report_id: int,
                                             author: User,
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
    update_report = CloudStorageUpdateReport.objects.select_related("storage").get(pk=update_report_id)
    cloud_storage = update_report.storage
    documents = NetworkDocument.objects.filter(pk__in=document_ids)

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
            executor.submit(download_and_process_file, doc, cloud_storage, author)
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

    # Обновление отчета
    if not failed:
        update_report.content["current_status"] = "Content loading completed successfully"
    else:
        update_report.content["current_status"] = "Content loading failed"

    update_report.save(update_fields=["content"])
    logger.info(f"Обработка завершена. Успешно: {len(success)}, Ошибки: {len(failed)}")

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