import time
from math import ceil

from celery import shared_task
from celery_progress.backend import ProgressRecorder

from app_sources.models import Document, CloudStorage, DocumentSourceType, StorageUpdateReport


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
def process_cloud_files(self, files: list[dict], cloud_storage: CloudStorage, update_report:StorageUpdateReport):
    total_counter = len(files)

    if total_counter == 0:
        return "Обработка завершена"

    errors_list = []
    progress_recorder = ProgressRecorder(self)
    progress_now, current = 0, 0
    progress_step = ceil(total_counter / 100)
    progress_description = f' Обрабатывается {total_counter} объектов'

    test = {'file_id': '"bb1f033d0bd79d63c0a7515d50f65f3d"',
     'file_name': 'Устав.pdf',
     'is_dir': False,
     'last_modified': 'Thu, 29 May 2025 05:54:28 GMT',
     'path': 'documents/Устав.pdf',
     'size': 12897578,
     'url': 'https://cloud.academydpo.org/public.php/webdav/documents/Устав.pdf'
     }
    result = {'new_files': [], 'updated_files': [], 'deleted_files': [], 'error': None}

    # Все документы из базы для данного хранилища
    db_documents = Document.objects.filter(
        cloud_storage=cloud_storage,
        source_type=DocumentSourceType.network.value
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


@shared_task(bind=True)
def download_file_content(documents: list[Document]):
    pass