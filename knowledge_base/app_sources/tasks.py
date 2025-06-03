import asyncio
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from math import ceil

from asgiref.sync import sync_to_async
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.core.files import File
from django.core.files.base import ContentFile
from django.utils import timezone

from app_sources.models import NetworkDocument, RawContent
from app_sources.storage_models import CloudStorage, CloudStorageUpdateReport
from utils.process_files import compute_sha512

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
def process_cloud_files(self, files: list[dict], cloud_storage: CloudStorage, update_report:CloudStorageUpdateReport):
    total_counter = len(files)

    if total_counter == 0:
        return "Обработка завершена"

    errors_list = []
    progress_recorder = ProgressRecorder(self)
    progress_now, current = 0, 0
    progress_step = ceil(total_counter / 100)
    progress_description = f' Обрабатывается {total_counter} объектов'

    # test = {'file_id': '"bb1f033d0bd79d63c0a7515d50f65f3d"',
    #  'file_name': 'Устав.pdf',
    #  'is_dir': False,
    #  'last_modified': 'Thu, 29 May 2025 05:54:28 GMT',
    #  'path': 'documents/Устав.pdf',
    #  'size': 12897578,
    #  'url': 'https://cloud.academydpo.org/public.php/webdav/documents/Устав.pdf'
    #  }

    result = {'new_files': [], 'updated_files': [], 'deleted_files': [], 'error': None}

    # Все документы из базы для данного хранилища
    db_documents = NetworkDocument.objects.filter(
        cloud_storage=cloud_storage,
    )
    db_urls_set = set(db_documents.values_list("url", flat=True))

    # Все urls, пришедшие из облака
    incoming_urls_set = set(file["url"] for file in files)

    # удалены в облаке
    deleted_urls_set = db_urls_set - incoming_urls_set
    result["deleted_files"] = list(
        db_documents.filter(url__in=deleted_urls_set).values_list("id", flat=True)
    )

    for file in files:
        current += 1
        url = file.get('url')
        if url in db_urls_set:
            result['updated_files'].append(file)
        else:
            result['new_files'].append(file)

        if current == (progress_now + 1) * progress_step:
            progress_now += 1
            progress_recorder.set_progress(progress_now, 100, description=progress_description)

    update_report.content["result"] = result
    update_report.content["status"] = "Sorting of synchronization files completed successfully"
    update_report.save()
    return "Обработка завершена"


def download_and_process_file(doc, cloud_storage):
    temp_path = None
    file_name = None
    print(f"{doc.url=}")
    try:
        # Получаем API для скачивания
        cloud_storage_api = cloud_storage.get_storage()

        # Скачиваем файл на диск
        temp_path, file_name = cloud_storage_api.download_file_to_disk_sync(doc.url)
    except Exception as e:
        doc.error_message=f"Ошибка при загрузке файла: {e}"
        doc.status = "er"
        doc.save()
        logger.error(f"[Document {doc.pk}] Ошибка при загрузке: {e}")
        return doc.pk, "fail"

    try:
        # Сохраняем файл в базу данных
        raw_content = RawContent.objects.create(document=doc)

        # with open(temp_path, 'rb') as f:
        #     raw_content.file.save(file_name, File(f), save=False)

        with open(temp_path, 'rb') as f:
            content = f.read()
        raw_content.file.save(file_name, BytesIO(content), save=False)

        raw_content.hash_content = compute_sha512(temp_path)
        raw_content.save()
        print(raw_content.url)
        doc.size = raw_content.file.size
        doc.synchronized_at = timezone.now()
        doc.status = "sy"  # synced
        doc.save()
    except Exception as e:
        doc.error_message = f"Ошибка при сохранении файла в БД: {e}"
        doc.status = "er"  # failed to save
        doc.save()
        logger.error(f"[Document {doc.pk}] Ошибка при сохранении: {e}")
        return doc.pk, "fail"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return doc.pk, "success"


@shared_task
def download_and_create_raw_content_parallel(document_ids: list[int], update_report_id: int, max_workers: int = 5):
    update_report = CloudStorageUpdateReport.objects.select_related("storage").get(pk=update_report_id)
    cloud_storage = update_report.storage
    documents = NetworkDocument.objects.filter(pk__in=document_ids)
    print(documents)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_and_process_file, doc, cloud_storage)
            for doc in documents
        ]
        for future in as_completed(futures):
            results.append(future.result())

    success = [pk for pk, status in results if status == "success"]
    failed = [pk for pk, status in results if status == "fail"]

    logger.info(f"Обработка завершена. Успешно: {len(success)}, Ошибки: {len(failed)}")


@shared_task
def download_and_create_raw_content(document_ids: list[int], update_report_id: int, max_workers: int = 5):
    update_report = CloudStorageUpdateReport.objects.select_related("storage").get(pk=update_report_id)
    cloud_storage = update_report.storage
    documents = NetworkDocument.objects.filter(pk__in=document_ids)
    print(documents)

    results = []
    for doc in documents:
        download_and_process_file(doc=doc, cloud_storage=cloud_storage)

    success = [pk for pk, status in results if status == "success"]
    failed = [pk for pk, status in results if status == "fail"]

    logger.info(f"Обработка завершена. Успешно: {len(success)}, Ошибки: {len(failed)}")