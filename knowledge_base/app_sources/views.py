import logging

from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render, redirect
from django.views import View
from django.views.generic import DetailView, ListView, CreateView

from app_sources.forms import CloudStorageForm
from app_sources.models import CloudStorage, Document

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
        print(f"{cloud_storage=}")
        synced_documents = request.POST.getlist("synced_documents")
        print(f"{cloud_storage=}")
        try:
            cloud_storage_api = cloud_storage.get_storage()  # Без проверки соединения
            logger.info(f"Успешно инициализировано хранилище: {cloud_storage.name}")
        except ValueError as e:
            logger.error(f"Ошибка инициализации хранилища {cloud_storage.name}: {e}")
            return render(request, 'app_sources/sync_result.html', {
                'result': {'error': f'Ошибка инициализации: {e}'},
                'cloud_storage': cloud_storage
            })

        result = {'new_files': [], 'updated_files': [], 'deleted_files': [], 'error': None}

        try:
            if synced_documents:
                server_files = cloud_storage_api.sync_selected(synced_documents)
            else:
                server_files = cloud_storage_api.sync_all()
        except Exception as e:
            logger.error(f"Ошибка получения файлов: {e}")
            result['error'] = f"Ошибка получения файлов: {e}"
            return render(request, 'app_sources/sync_result.html', {
                'result': result,
                'cloud_storage': cloud_storage
            })

        server_paths = {item['path'] for item in server_files}

        for item in server_files:
            try:
                document, created = Document.objects.get_or_create(
                    cloud_storage=cloud_storage,
                    path=item['path'],
                    defaults={'title': item['file_name']}
                )
                remote_updated = parse_datetime(item['last_modified']) if item['last_modified'] else None
                if created or (remote_updated and remote_updated > document.remote_updated) or document.metadata.get('etag') != item['etag']:
                    document.file_id = item.get('fileid')
                    document.remote_updated = remote_updated
                    document.metadata['etag'] = item['etag'] or ''
                    document.metadata['webdav_path'] = item['path']
                    document.save()

                    file_content = cloud_storage_api.download_file(item['path'])
                    raw_content, _ = RawContent.objects.get_or_create(source=document)
                    raw_content.file.save(item['file_name'], ContentFile(file_content))
                    raw_content.hash_content = raw_content.generate_hash()
                    raw_content.save()
                    document.local_updated = remote_updated
                    document.save()

                    (result['new_files'] if created else result['updated_files']).append(item['path'])
                    logger.info(f"{'Создан' if created else 'Обновлен'} документ: {item['path']}")
            except Exception as e:
                logger.error(f"Ошибка обработки файла {item['path']}: {e}")
                result['error'] = f"Ошибка обработки файла {item['path']}: {e}"
                break

        if not synced_documents and not result['error']:
            # Удаляем локальные документы, отсутствующие на сервере
            local_documents = Document.objects.filter(cloud_storage=cloud_storage)
            for doc in local_documents:
                if doc.path not in server_paths:
                    result['deleted_files'].append(doc.path)
                    doc.delete()  # Или: doc.is_deleted = True; doc.save()
                    logger.info(f"Удалён документ: {doc.path}")

        return render(request, 'app_sources/sync_result.html', {
            'result': result,
            'cloud_storage': cloud_storage
        })


