import logging
from pprint import pprint

from dateutil.parser import parse
from django.contrib.auth.mixins import UserPassesTestMixin
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse_lazy
from django.utils import timezone
from django.views import View
from django.views.generic import DetailView, ListView, CreateView, UpdateView, DeleteView

from app_core.models import KnowledgeBase
from app_sources.forms import CloudStorageForm
from app_sources.models import NetworkDocument
from app_sources.storage_models import CloudStorage, CloudStorageUpdateReport, Storage
from app_sources.tasks import process_cloud_files, download_and_create_raw_content

logger = logging.getLogger(__name__)


class StoragePermissionMixin(UserPassesTestMixin):
    """
    Mixin для проверки прав доступа к хранилищу:
    доступ разрешён только владельцу связанной базы знаний (KnowledgeBase) или суперпользователю.
    """

    def test_func(self):
        storage = self.get_object()
        kb = getattr(storage, "knowledge_base", None)
        return kb and (kb.owner == self.request.user or self.request.user.is_superuser)


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

    def get_initial(self):
        """Предзаполненные значения для удобства при тестировании"""
        return {
            "url": "https://cloud.academydpo.org/public.php/webdav/",
            "root_path": "documents/",
            "auth_type": "token",
            "token": "rqJWt7LzPGKcyNw"
        }

    def dispatch(self, request, *args, **kwargs):
        """Сохраняем knowledge_base для дальнейшего использования"""
        self.knowledge_base = get_object_or_404(KnowledgeBase, pk=kwargs['kb_pk'])
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        """Установка связи с базой знаний перед валидацией"""
        kb_pk = self.kwargs.get("kb_pk")
        if not kb_pk:
            form.add_error(None, "Не передан ID базы знаний")
            return self.form_invalid(form)

        form.instance.kb = self.knowledge_base
        form.instance.author = self.request.user
        return super().form_valid(form)

    def get_form(self, form_class=None):
        form = super().get_form(form_class)
        kb_pk = self.kwargs.get("kb_pk")
        if kb_pk:
            form.instance.kb_id = kb_pk
        form.instance.author = self.request.user
        return form

    # def get_context_data(self, **kwargs):
    #     """Добавить ID базы знаний в контекст (если нужно для шаблона)"""
    #     context = super().get_context_data(**kwargs)
    #     context['kb_pk'] = self.kwargs.get('kb_pk')
    #     return context


class CloudStorageUpdateView(UpdateView):
    pass


class CloudStorageDeleteView(DeleteView):
    model = CloudStorage
    success_url = reverse_lazy("sources:cloudstorage_list")


class CloudStorageSyncView(View):
    """
    Синхронизация облачного хранилища: получение списка файлов, создание отчёта, запуск фоновой задачи.
    """

    def post(self, request, pk):
        cloud_storage = get_object_or_404(CloudStorage, pk=pk)
        synced_documents = request.POST.getlist("synced_documents")
        storage_update_report = CloudStorageUpdateReport.objects.create(storage=cloud_storage)

        try:
            cloud_storage_api = cloud_storage.get_storage()
            logger.info(f"Хранилище инициализировано: {cloud_storage.name}")
            storage_update_report.content["current_status"] = "api successfully initialized"
        except ValueError as e:
            logger.error(f"Ошибка инициализации: {e}")
            storage_update_report.content["current_status"] = "api initialization failed"
            storage_update_report.content.setdefault("errors", []).append(str(e))
            storage_update_report.save()
            return render(request, 'app_sources/sync_result.html', {
                'result': {'error': f'Ошибка инициализации: {e}'},
                'cloud_storage': cloud_storage
            })

        storage_update_report.save()

        try:
            if synced_documents:
                storage_update_report.content["sync_type"] = "custom"
                storage_files = []  # cloud_storage_api.sync_selected(synced_documents)
            else:
                storage_update_report.content["sync_type"] = "all"
                storage_files = cloud_storage_api.list_directory(path=cloud_storage_api.root_path)
                pprint(storage_files)

            storage_update_report.content["current_status"] = "file list retrieved"
            storage_update_report.save()

            task = process_cloud_files.delay(
                files=storage_files,
                cloud_storage=cloud_storage,
                update_report=storage_update_report
            )

            return render(request, 'app_sources/cloudstorage_progress_report.html', {
                'task_id': task.task_id,
                'cloudstorage': cloud_storage,
                "next_step_url": reverse_lazy("sources:storageupdatereport_detail", args=[storage_update_report.id])
            })

        except Exception as e:
            logger.exception("Ошибка синхронизации файлов")
            storage_update_report.content["current_status"] = "failed to get file list from cloud"
            storage_update_report.content.setdefault("errors", []).append(str(e))
            storage_update_report.save()
            return render(request, 'app_sources/sync_result.html', {
                'result': {'error': f"Ошибка получения файлов: {e}"},
                'cloud_storage': cloud_storage
            })


class StorageUpdateReportDetailView(DetailView):
    """Детальный просмотр отчёта о синхронизации облачного хранилища"""
    model = CloudStorageUpdateReport


class NetworkDocumentsMassCreateView(View):
    """
    Создание документов (NetworkDocument) на основе отчёта синхронизации (CloudStorageUpdateReport).
    """

    def post(self, request, pk):
        update_report = get_object_or_404(CloudStorageUpdateReport, pk=pk)
        new_files = update_report.content.get("result", {}).get("new_files", [])

        db_documents = NetworkDocument.objects.filter(cloud_storage=update_report.storage)
        existing_urls = set(db_documents.values_list("url", flat=True))
        new_urls = {f["url"] for f in new_files}
        duplicates = existing_urls & new_urls

        bulk_container = []
        for f in new_files:
            if f["url"] in duplicates:
                f["create_status"] = "already_exists"
                continue
            try:
                remote_updated = parse(f.get("last_modified", ''))
            except Exception:
                remote_updated = None
            bulk_container.append(NetworkDocument(
                cloud_storage=update_report.storage,
                path=f["path"],
                file_id=f["file_id"],
                size=f["size"],
                url=f["url"],
                remote_updated=remote_updated,
                synchronized_at=timezone.now(),
            ))

        if bulk_container:
            created_docs = NetworkDocument.objects.bulk_create(bulk_container)
            created_ids = [doc.id for doc in created_docs]
            update_report.content["created_docs"] = created_ids
            for f in new_files:
                if f.get("create_status") != "already_exists":
                    f["create_status"] = "created"
            update_report.content["status"] = "Documents successfully created, download content in progress..."
            update_report.save()

            download_and_create_raw_content.delay(
                document_ids=created_ids,
                update_report_id=update_report.pk
            )

        return redirect(reverse_lazy("sources:storageupdatereport_detail", args=[pk]))


class LocalStorageListView(ListView):
    """Списковый просмотр объектов модели Локальное хранилище"""
    model = Storage


class LocalStorageDetailView(StoragePermissionMixin, DetailView):
    """Детальный просмотр объекта модели Локальное хранилище (с проверкой прав доступа)"""
    model = Storage


class LocalStorageCreateView(CreateView):
    """Создание объекта модели Локальное хранилище"""
    model = Storage
    fields = "__all__"
