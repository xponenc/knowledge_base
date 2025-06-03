import logging
from pprint import pprint

from dateutil.parser import parse
from django.utils import timezone
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import DetailView, ListView, CreateView

from app_sources.forms import CloudStorageForm
from app_sources.tasks import process_cloud_files, download_and_create_raw_content_parallel, \
    download_and_create_raw_content
from app_sources.models import CloudStorage, Document, StorageUpdateReport, DocumentSourceType

logger = logging.getLogger(__name__)


class CloudStorageDetailView(DetailView):
    """Детальный просмотр объекта модели Облачное хранилище"""

    model = CloudStorage


class CloudStorageListView(ListView):
    """Списковый просмотр объектов модели Облачное хранилище"""

    model = CloudStorage


class CloudStorageCreateView(CreateView):
    """Создание объекта модели Облачное хранилище"""

    model = CloudStorage
    form_class = CloudStorageForm

    # def post(self, request, *args, **kwargs):
    #     form = CloudStorageForm(request)
    #     return render(request, "app_sources/cloudstorage_form.html", context)


class CloudStorageSyncView(View):
    """Синхронизация файлов облачного хранилища с локальной копией"""

    def post(self, request, pk):
        try:
            cloud_storage = CloudStorage.objects.get(pk=pk)
        except CloudStorage.DoesNotExist:
            logger.error(f"CloudStorage с ID {pk} не найден")
            return render(request, 'app_sources/sync_result.html', {
                'result': {'error': 'CloudStorage не найден'},
                'cloud_storage': None
            })
        synced_documents = request.POST.getlist("synced_documents")
        storage_update_report = StorageUpdateReport.objects.create(storage=cloud_storage)
        try:
            cloud_storage_api = cloud_storage.get_storage()
            logger.info(f"Успешно инициализировано хранилище: {cloud_storage.name}")
            storage_update_report.content["current_status"] = "api successfully initialized"
            storage_update_report.save()
        except ValueError as e:
            logger.error(f"Ошибка инициализации хранилища {cloud_storage.name}: {e}")
            storage_update_report.content["current_status"] = "api successfully failed"
            storage_update_report.save()
            return render(request, 'app_sources/sync_result.html', {
                'result': {'error': f'Ошибка инициализации: {e}'},
                'cloud_storage': cloud_storage
            })

        result = {'new_files': [], 'updated_files': [], 'deleted_files': [], 'error': None}

        storage_files = []
        try:
            if synced_documents:
                storage_update_report.content["sync_type"] = "custom"
                storage_update_report.save()
                # server_files = cloud_storage_api.sync_selected(synced_documents)
                pass
            else:
                storage_update_report.content["sync_type"] = "all"
                storage_update_report.save()
                storage_files = cloud_storage_api.list_directory(path=cloud_storage_api.root_path)

                pprint(storage_files)
        except Exception as e:
            logger.error(f"Ошибка получения файлов: {e}")
            storage_update_report.content["current_status"] = "failed to get file list from cloud"
            storage_update_report.content.setdefault("errors", []).append(e)
            storage_update_report.save()
            result['error'] = f"Ошибка получения файлов: {e}"
            return render(request, 'app_sources/sync_result.html', {
                'result': result,
                'cloud_storage': cloud_storage
            })
        if storage_files:
            storage_update_report.content["current_status"] = "successfully retrieved file list from cloud"
            storage_update_report.save()
            # Запускаем фоновую обработку загруженных объектов
            celery_process_cloud_files_result = process_cloud_files.delay(files=storage_files,
                                                                          cloud_storage=cloud_storage,
                                                                          update_report=storage_update_report)
            return render(request, 'app_sources/cloudstorage_progress_report.html',
                          context={'task_id': celery_process_cloud_files_result.task_id,
                                   'cloudstorage': cloud_storage,
                                   "next_step_url": reverse_lazy("sources:storageupdatereport_detail",
                                                                 args=[storage_update_report.id, ]), })
        server_paths = {item['path'] for item in storage_files}
        print(server_paths)

        # for item in server_files:
        #     try:
        #         document, created = Document.objects.get_or_create(
        #             cloud_storage=cloud_storage,
        #             path=item['path'],
        #             defaults={'title': item['file_name']}
        #         )
        #         remote_updated = parse_datetime(item['last_modified']) if item['last_modified'] else None
        #         if created or (remote_updated and remote_updated > document.remote_updated) or document.metadata.get('etag') != item['etag']:
        #             document.file_id = item.get('fileid')
        #             document.remote_updated = remote_updated
        #             document.metadata['etag'] = item['etag'] or ''
        #             document.metadata['webdav_path'] = item['path']
        #             document.save()
        #
        #             file_content = cloud_storage_api.download_file(item['path'])
        #             raw_content, _ = RawContent.objects.get_or_create(source=document)
        #             raw_content.file.save(item['file_name'], ContentFile(file_content))
        #             raw_content.hash_content = raw_content.generate_hash()
        #             raw_content.save()
        #             document.local_updated = remote_updated
        #             document.save()
        #
        #             (result['new_files'] if created else result['updated_files']).append(item['path'])
        #             logger.info(f"{'Создан' if created else 'Обновлен'} документ: {item['path']}")
        #     except Exception as e:
        #         logger.error(f"Ошибка обработки файла {item['path']}: {e}")
        #         result['error'] = f"Ошибка обработки файла {item['path']}: {e}"
        #         break
        #
        # if not synced_documents and not result['error']:
        #     # Удаляем локальные документы, отсутствующие на сервере
        #     local_documents = Document.objects.filter(cloud_storage=cloud_storage)
        #     for doc in local_documents:
        #         if doc.path not in server_paths:
        #             result['deleted_files'].append(doc.path)
        #             doc.delete()  # Или: doc.is_deleted = True; doc.save()
        #             logger.info(f"Удалён документ: {doc.path}")

        return render(request, 'app_sources/sync_result.html', {
            'result': result,
            'cloud_storage': cloud_storage
        })


class StorageUpdateReportDetailView(DetailView):
    """Детальный просмотр объекта Отчет о синхронизации облачного хранилища"""

    model = StorageUpdateReport

    # def get_context_data(self, **kwargs):
    #     context = super(StorageUpdateReportDetailView, self).get_context_data(**kwargs)
    #     report = self.object
    #     created_docs_ids = report.content.get("created_docs")
    #     if created_docs_ids:
    #         created_documents = Document.objects.filter(id__in=created_docs_ids)
    #         report.content["created_docs"] =


class DocumentsMassCreateView(View):
    """Создание новых объектов Document из отчета по синхронизации"""
    def post(self, request, pk):
        update_report = get_object_or_404(StorageUpdateReport, pk=pk)
        new_files = update_report.content.get("result", {}).get("new_files", [])

        # Все документы из базы для данного хранилища
        db_documents = Document.objects.filter(
            cloud_storage=update_report.storage,
            source_type=DocumentSourceType.network.value
        )
        db_urls_set = set(db_documents.values_list("url", flat=True))
        new_urls_set = set(file["url"] for file in new_files)

        # Проверка на пересечение
        duplicated_urls = db_urls_set & new_urls_set

        if duplicated_urls:
            print(duplicated_urls)

        bulk_container = []
        for new_file in new_files:
            if new_file["url"] in duplicated_urls:
                new_file["create_status"] = "always_exist"
                continue
            # test = {'file_id': '"bb1f033d0bd79d63c0a7515d50f65f3d"',
            #         'file_name': 'Устав.pdf',
            #         'is_dir': False,
            #         'last_modified': 'Thu, 29 May 2025 05:54:28 GMT',
            #         'path': 'documents/Устав.pdf',
            #         'size': 12897578,
            #         'url': 'https://cloud.academydpo.org/public.php/webdav/documents/Устав.pdf'
            #         }
            try:
                remote_updated = parse(new_file.get('last_modified', ''))
            except Exception as e:
                remote_updated = None
            document = Document(
                cloud_storage=update_report.storage,
                path=new_file["path"],
                file_id=new_file["file_id"] ,
                size=new_file["size"],
                url=new_file["url"],
                remote_updated = remote_updated,
                synchronized_at = timezone.now(),
            )
            bulk_container.append(document)
        if bulk_container:
            created_docs = Document.objects.bulk_create(bulk_container)
            created_ids = [doc.id for doc in created_docs]
            update_report.content["created_docs"] = created_ids
            for new_file in new_files:
                if new_file.get("create_status", "") != "always_exist":
                    new_file["create_status"] = "created"
            update_report.content["status"] = "Documents successfully created, download content in progress..."
            update_report.save()
            # Запуск фоновой задачи скачивания контента и создания RawContent для созданных Document
            print("# Запуск фоновой задачи скачивания контента и создания RawContent для созданных Document")
            download_and_create_raw_content.delay(
                document_ids=created_ids, update_report_id=update_report.pk
            )

        return redirect(reverse_lazy("sources:storageupdatereport_detail", args=[pk, ]))

