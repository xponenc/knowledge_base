import json
import logging
import os

from django import forms
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.core.paginator import Paginator
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import Func, IntegerField, F, Q, Prefetch, Count
from django.db.models.fields.json import KeyTextTransform
from django.db.models.functions import Cast
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render, redirect
from django.template.loader import render_to_string
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import DetailView, ListView, CreateView, UpdateView, DeleteView

from app_chunks.models import Chunk, ChunkStatus
from app_core.models import KnowledgeBase
from app_sources.content_models import RawContent, ContentStatus
from app_sources.forms import CloudStorageForm
from app_sources.models import HierarchicalContextMixin
from app_sources.report_models import CloudStorageUpdateReport
from app_sources.services.google_sheets_manager import GoogleSheetsManager
from app_sources.source_models import URL, NetworkDocument, LocalDocument, SourceStatus
from app_sources.storage_forms import StorageTagsForm, StorageScanTagsForm, StorageScanParamForm
from app_sources.storage_models import CloudStorage, LocalStorage, URLBatch, WebSite
from knowledge_base.settings import BASE_DIR
from app_sources.tasks import process_cloud_files

logger = logging.getLogger(__name__)


class StoragePermissionMixin(UserPassesTestMixin):
    """
    Mixin для проверки прав доступа к хранилищу:
    доступ разрешён только владельцу связанной базы знаний (KnowledgeBase) или суперпользователю.
    """

    def test_func(self):
        if self.request.user.is_superuser:
            return True
        storage = self.get_object()
        if not storage:
            return False
        kb = getattr(storage, "knowledge_base", None)
        return kb and kb.owner == self.request.user


class StorageTagsView(LoginRequiredMixin, StoragePermissionMixin, View):
    """Просмотр и управление облаком тегов хранилища"""
    STORAGE_MODELS = {
        "cloud": CloudStorage,
        "local": LocalStorage,
        "website": WebSite,
        "urlbatch": URLBatch,
    }

    def get_object(self):
        storage_type = self.kwargs.get("storage_type")
        storage_pk = self.kwargs.get("storage_pk")
        storage_class = self.STORAGE_MODELS.get(storage_type)
        if not storage_class:
            return None
        try:
            return storage_class.objects.get(pk=storage_pk)
        except (storage_class.DoesNotExist, ValueError):
            return None

    @staticmethod
    def get_longest_tags(storage):
        """Поиск самой длинной цепочки тегов у источника. Версия для PostgreSQL"""
        if isinstance(storage, WebSite):
            qs = URL.objects.filter(site=storage)
        elif isinstance(storage, URLBatch):
            qs = URL.objects.filter(batch=storage)
        elif isinstance(storage, CloudStorage):
            qs = NetworkDocument.objects.filter(storage=storage)
        elif isinstance(storage, LocalStorage):
            qs = LocalDocument.objects.filter(storage=storage)
        else:
            return []

        return (
                qs.annotate(
                    tag_count=Cast(Func(F('tags'), function='jsonb_array_length'), IntegerField())
                )
                .order_by('-tag_count')
                .values_list('tags', flat=True)
                .first() or []
        )

    def get(self, request, storage_type, *args, **kwargs):
        storage = self.get_object()
        if not storage:
            return render(request, "errors/404.html", status=404)

        storage_tags_update_form = StorageTagsForm(instance=storage)
        storage_tags_scan_form = StorageScanTagsForm()

        # Поиск источника с самыми длинными tags

        if isinstance(storage, WebSite):
            tag_lists = list(URL.objects.filter(site=storage).values_list("urlcontent__tags", flat=True))
        elif isinstance(storage, URLBatch):
            tag_lists = list(URL.objects.filter(batch=storage).values_list("urlcontent__tags", flat=True))
        elif isinstance(storage, CloudStorage):
            tag_lists = list(NetworkDocument.objects.filter(storage=storage).values_list("tags", flat=True))
        elif isinstance(storage, LocalStorage):
            tag_lists = list(LocalDocument.objects.filter(storage=storage).values_list("tags", flat=True))
        else:
            tag_lists = []
        longest_tags = max(tag_lists, key=lambda tags: len(tags) if tags else 0, default=[])

        # Версия для PostgreSQL
        # longest_tags = self.get_longest_tags(storage)

        context = {
            "storage": storage,
            "storage_type": storage_type,
            "longest_tags": longest_tags,
            "storage_tags_update_form": storage_tags_update_form,
            "storage_tags_scan_form": storage_tags_scan_form,
        }

        return render(
            request=request,
            template_name="app_sources/storage_tags.html",
            context=context,
        )

    def post(self, request, storage_type, *args, **kwargs):
        storage = self.get_object()
        if not storage:
            return render(request, "errors/404.html", status=404)
        # storage.tags = []
        # storage.save()
        storage_tags_update_form = StorageTagsForm(request.POST, instance=storage)
        storage_tags_scan_form = StorageScanTagsForm(request.POST)

        # Поиск источника с самыми длинными tags

        if isinstance(storage, WebSite):
            tag_lists = list(URL.objects.filter(site=storage).values_list("urlcontent__tags", flat=True))
        elif isinstance(storage, URLBatch):
            tag_lists = list(URL.objects.filter(batch=storage).values_list("urlcontent__tags", flat=True))
        elif isinstance(storage, CloudStorage):
            tag_lists = list(NetworkDocument.objects.filter(storage=storage).values_list("tags", flat=True))
        elif isinstance(storage, LocalStorage):
            tag_lists = list(LocalDocument.objects.filter(storage=storage).values_list("tags", flat=True))
        else:
            tag_lists = []
        longest_tags = max(tag_lists, key=lambda tags: len(tags) if tags else 0, default=[])

        # Версия для PostgreSQL
        # longest_tags = self.get_longest_tags(storage)

        context = {
            "storage": storage,
            "storage_type": storage_type,
            "storage_tags_update_form": storage_tags_update_form,
            "storage_tags_scan_form": storage_tags_scan_form,
            "longest_tags": longest_tags,
        }

        updated = False

        if storage_tags_update_form.is_valid():
            if storage_tags_update_form.changed_data:
                storage_tags_update_form.save()
                updated = True

        if storage_tags_scan_form.is_valid():
            max_depth = storage_tags_scan_form.cleaned_data.get("scanning_depth") or 0

            if isinstance(storage, WebSite):
                tag_lists = list(URL.objects.filter(site=storage).values_list("urlcontent__tags", flat=True))
            elif isinstance(storage, URLBatch):
                tag_lists = list(URL.objects.filter(batch=storage).values_list("urlcontent__tags", flat=True))
            elif isinstance(storage, CloudStorage):
                tag_lists = list(NetworkDocument.objects.filter(storage=storage).values_list("tags", flat=True))
            elif isinstance(storage, LocalStorage):
                tag_lists = list(LocalDocument.objects.filter(storage=storage).values_list("tags", flat=True))
            else:
                tag_lists = []

            # Сбор уникальных тегов, ограниченных по max_depth
            new_tags = set()
            for tags in tag_lists:
                if isinstance(tags, list):
                    new_tags.update(tags[:max_depth])

            existing_tags = storage.tags if isinstance(storage.tags, list) else []
            combined_tags = set(existing_tags).union(new_tags)
            sorted_tags = sorted(combined_tags, reverse=True)

            storage.tags = sorted_tags
            storage.save()
            updated = True

        if updated:
            return redirect("sources:storage_tags", storage_type=storage_type, storage_pk=storage.pk)

        # Если ни одна форма невалидна — отобразим ошибки
        return render(
            request,
            template_name="app_sources/storage_tags.html",
            context=context,
        )


class CloudStorageDetailView(LoginRequiredMixin, StoragePermissionMixin, HierarchicalContextMixin, DetailView):
    """Детальный просмотр объекта модели Облачное хранилище"""
    model = CloudStorage
    #
    # def update_document_descriptions_from_csv(self, csv_path):
    #     csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path )
    #     with open(csv_path, newline='', encoding='utf-8') as csvfile:
    #         reader = csv.DictReader(csvfile, delimiter=',')
    #         updated = 0
    #         not_found = []
    #
    #         for row in reader:
    #             title = row.get("Название файла", "").strip()
    #             description = row.get("Название документа", "").strip()
    #
    #             if not title:
    #                 continue  # пропускаем строки без названия файла
    #
    #             try:
    #                 doc = NetworkDocument.objects.get(title=title)
    #                 doc.description = description
    #                 doc.save()
    #                 updated += 1
    #             except NetworkDocument.DoesNotExist:
    #                 not_found.append(title)
    #
    #     print(f"Обновлено документов: {updated}")
    #     if not_found:
    #         print("Не найдены документы с названиями файлов:")
    #         for title in not_found:
    #             print(f" - {title}")

    def get(self, request, *args, **kwargs):
        self.object = self.get_object()
        context = self.get_context_data()
        # self.update_document_descriptions_from_csv("DocScanner_Summary - Лист1.csv")
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            html = render_to_string(
                "app_sources/include/network_documents_page.html",
                context,
                request=request
            )
            return JsonResponse({"html": html})

        return self.render_to_response(context)

    def get_context_data(self, **kwargs):
        network_storage = self.object
        context = super().get_context_data()

        # Распределение чанков по размеру в токенах, должно быть поле metadata.size_in_tokens
        chunk_distribution = (
            Chunk.objects
            .filter(
                Q(raw_content__network_document__storage=network_storage) |
                Q(cleaned_content__raw_content__network_document__storage=network_storage),
                status=ChunkStatus.ACTIVE.value)
            .annotate(size=Cast(KeyTextTransform('size_in_tokens', 'metadata'), IntegerField()))
            .values('size')
            .annotate(count=Count('id'))
            .order_by('size')
        )


        # Распределение URL по статусу
        document_status_map = {s.value: s.display_name for s in SourceStatus}
        status_counts = (NetworkDocument.objects.filter(storage=network_storage)
                         .values('status')
                         .annotate(count=Count('id'))
                         .order_by('status')
                         )

        document_distribution = [
            {
                "status": document_status_map.get(row["status"], row["status"]),
                "count": row["count"]
            }
            for row in status_counts
        ]

        # отчеты по обновлениям
        update_report_count = CloudStorageUpdateReport.objects.filter(storage=network_storage).count()
        context["update_report_count"] = update_report_count
        # Передаём три последних отчёта
        update_reports_last = (CloudStorageUpdateReport.objects
                               .select_related("author")
                               .filter(storage=network_storage)
                               .defer("storage", "content", "running_background_tasks")
                               .order_by("-created_at")[:3])
        context["update_reports_last"] = update_reports_last

        source_statuses = [
            (status.value, status.display_name) for status in SourceStatus
        ]

        filters = {
            "status": source_statuses
        }

        sorting_list = [
            ("-created_at", "дата создания по возрастанию"),
            ("created_at", "дата создания по убыванию"),
            ("-title", "имя по возрастанию"),
            ("title", "имя по убыванию"),
            ("-url", "url по возрастанию"),
            ("url", "url по убыванию"),
        ]

        # Получаем параметры поиска и фильтрации из запроса
        search_query = self.request.GET.get("search", "").strip()
        status_filter = self.request.GET.getlist("status", None)
        sorting = self.request.GET.get("sorting", None)

        network_documents = network_storage.documents.order_by("title")

        if search_query:
            network_documents = network_documents.filter(
                Q(title__icontains=search_query) |
                Q(url__icontains=search_query) |
                Q(description__icontains=search_query)
            )

        if status_filter:
            valid_statuses = [status.value for status in SourceStatus]
            # Оставляем только корректные значения
            status_filter = [s for s in status_filter if s in valid_statuses]
            if status_filter:
                network_documents = network_documents.filter(status__in=status_filter)

        if sorting:
            valid_sorting = [value for value, name in sorting_list]
            if sorting in valid_sorting:
                network_documents = network_documents.order_by(sorting)

        rawcontent_qs = RawContent.objects.filter(
            status=ContentStatus.READY.value
        ).select_related(
            'cleanedcontent'
        ).order_by("-created_at")[:1]

        # Prefetch rawcontent_set → в each networkdocument.related_rawcontents будет список из 0 или 1
        network_documents = network_documents.select_related(
            "report__storage", "storage"
        ).prefetch_related(
            Prefetch('rawcontent_set', queryset=rawcontent_qs, to_attr='related_rawcontents')
        ).annotate(
            rawcontent_total_count=Count('rawcontent')
        )

        paginator = Paginator(network_documents, 3)
        page_number = self.request.GET.get("page")
        page_obj = paginator.get_page(page_number)

        # Назначаем явно (0 или 1 элемент)
        for doc in page_obj.object_list:
            doc.active_rawcontent = doc.related_rawcontents[0] if doc.related_rawcontents else None

        # Формируем query string без page
        query_params = self.request.GET.copy()
        query_params.pop('page', None)
        paginator_query = query_params.urlencode()
        if paginator_query:
            paginator_query += '&'

        context["page_obj"] = page_obj
        context["paginator_query"] = paginator_query
        context["paginator"] = paginator
        context["documents"] = page_obj.object_list
        context["is_paginated"] = page_obj.has_other_pages()
        context["search_query"] = search_query
        context["status_filter"] = status_filter
        context["source_statuses"] = source_statuses
        context["filters"] = filters
        context["sorting_list"] = sorting_list
        context["chunk_distribution"] = json.dumps(list(chunk_distribution), cls=DjangoJSONEncoder)
        context["source_distribution"] = json.dumps(list(document_distribution), cls=DjangoJSONEncoder)
        return context


class CloudStorageListView(LoginRequiredMixin, ListView):
    """Списковый просмотр объектов модели Облачное хранилище"""
    model = CloudStorage

    def get_queryset(self):
        queryset = super().get_queryset()
        if self.request.user.is_superuser:
            return queryset
        queryset = queryset.filter(soft_deleted_at__isnull=True).filter(owners=self.request.user)
        return queryset


class CloudStorageCreateView(LoginRequiredMixin, StoragePermissionMixin, CreateView):
    """Создание объекта модели Облачное хранилище"""
    model = CloudStorage
    form_class = CloudStorageForm

    def get_initial(self):
        """Предзаполненные значения для удобства при тестировании"""
        return {
            "name": "Облако Академии ДПО",
            "api_type": "webdav",
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


class CloudStorageUpdateView(LoginRequiredMixin, UpdateView):
    model = CloudStorage
    fields = ("name", "description")

    def get_form(self, form_class=None):
        form = super().get_form(form_class)
        form.fields["name"].widget.attrs.update({
            "class": "custom-field__input custom-field__input_wide",
            "placeholder": "Введите название базы знаний"
        })

        form.fields["description"].widget = forms.Textarea(attrs={
            "class": "custom-field__input custom-field__input_wide custom-field__input_textarea",
            "placeholder": "Добавьте описание (необязательно)"
        })
        return form


class CloudStorageDeleteView(LoginRequiredMixin, DeleteView):
    model = CloudStorage
    success_url = reverse_lazy("sources:cloudstorage_list")


class CloudStorageSyncView(LoginRequiredMixin, StoragePermissionMixin, View):
    """
    Синхронизация облачного хранилища: получение списка файлов, создание отчёта, запуск фоновой задачи.
    """

    def get(self, request, pk):
        cloud_storage = CloudStorage.objects.get(pk=pk)
        scan_params_form = StorageScanParamForm()
        context = {
            "form": scan_params_form,
            "storage": cloud_storage,
            "storage_type_eng": "cloud",
            "storage_type_ru": "Облако",
        }
        return render(request=request,
                      template_name="app_sources/storage_sync.html",
                      context=context
                      )

    def post(self, request, pk):

        cloud_storage = get_object_or_404(CloudStorage, pk=pk)
        kb = cloud_storage.kb
        scan_params_form = StorageScanParamForm(request.POST)
        if scan_params_form.is_valid():
            recognize_content = scan_params_form.cleaned_data.get("recognize_content", False)
            do_summarization =  scan_params_form.cleaned_data.get("do_summarization", False)
        else:
            recognize_content = False
            do_summarization = False

        synced_documents = request.POST.getlist("synced_documents")
        logger.info(f"Начало синхронизации хранилища: {cloud_storage.name}, запущена {request.user}")
        storage_update_report = CloudStorageUpdateReport.objects.create(storage=cloud_storage, author=self.request.user)
        try:
            task = process_cloud_files.delay(
                files=synced_documents,
                update_report_pk=storage_update_report.pk,
                recognize_content=recognize_content,
                do_summarization=do_summarization,
            )

            storage_update_report.running_background_tasks[task.id] = "Синхронизация файлов"
            storage_update_report.save(update_fields=["running_background_tasks", ])

            context = {
                "kb": kb,
                "task_id": task.id,
                "task_name": f"Синхронизация хранилища {cloud_storage._meta.verbose_name} {cloud_storage.name}",
                "task_object_url": storage_update_report.get_absolute_url(),
                "task_object_name": "Отчет о векторизации",
                "next_step_url": storage_update_report.get_absolute_url(),
            }
            return render(request=request,
                          template_name="celery_task_progress.html",
                          context=context
                          )


        except Exception as e:
            logger.exception(f"Ошибка синхронизации хранилища: {cloud_storage.name}, запущена {request.user},"
                             f" отчет [CloudStorageUpdateReport id {storage_update_report.pk}]")
            storage_update_report.content.setdefault("errors", []).append(str(e))
            storage_update_report.save()
            return render(request, 'app_sources/sync_result.html', {
                'result': {'error': f"Ошибка получения файлов: {e}"},
                'cloud_storage': cloud_storage
            })


class CloudStorageExportGoogleSheetView(LoginRequiredMixin, StoragePermissionMixin, View):
    """Экспорт данных облачного хранилища в GoogleSheets"""

    def get(self, request, pk, *args, **kwargs):
        pass

    def post(self, request, pk, *args, **kwargs):
        cloud_storage = get_object_or_404(CloudStorage, pk=pk)
        cloud_storage_name = cloud_storage.name
        network_documents = cloud_storage.network_documents.all()
        credentials_file = os.path.join(BASE_DIR, "product_config", "credentials.json")

        google_sheets_manager = GoogleSheetsManager(
            credentials_file=credentials_file,
            short_sheet_name=f"{cloud_storage_name}_DocScanner_Summary",
            full_sheet_name=f"{cloud_storage_name}_DocScanner_FullSummary"
        )

        google_sheets_manager.export_short_summary(request=request, export_data=network_documents)

        return HttpResponse(google_sheets_manager.short_shared_link)