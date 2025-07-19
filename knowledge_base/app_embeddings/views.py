from django import forms
from django.contrib.auth.mixins import PermissionRequiredMixin, LoginRequiredMixin
from django.core.paginator import Paginator
from django.db.models import Count
from django.shortcuts import get_object_or_404, render
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView

from app_core.models import KnowledgeBase
from app_embeddings.models import EmbeddingEngine, EmbeddingsReport
from app_sources.storage_models import WebSite, CloudStorage, LocalStorage, URLBatch
from app_sources.storage_views import StoragePermissionMixin
from app_embeddings.tasks import create_vectors_task, universal_create_vectors_task, test_task
from utils.setup_logger import setup_logger

logger = setup_logger(name=__file__, log_dir="logs/embeddings", log_file="embeddings.log")


class EngineListView(LoginRequiredMixin, PermissionRequiredMixin, ListView):
    """Списковый просмотр EmbeddingEngine"""
    model = EmbeddingEngine
    paginate_by = 10
    permission_required = "app_embeddings.view_embeddingengine"
    permission_denied_message = "У вас нет прав для просмотра списка объектов модели"
    template_name = "app_embeddings/engine_list.html"

    def get_queryset(self):
        return EmbeddingEngine.objects.annotate(
            bases_count=Count("bases", distinct=True)
        )


class EngineDetailView(LoginRequiredMixin, PermissionRequiredMixin, DetailView):
    """Детальный просмотр EmbeddingEngine"""
    model = EmbeddingEngine
    queryset = EmbeddingEngine.objects.select_related("author")
    permission_required = "app_embeddings.view_embeddingengine"
    permission_denied_message = "У вас нет прав для просмотра объекта"
    template_name = "app_embeddings/engine_detail.html"
    paginate_bases_by = 10

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        base_qs = (
            KnowledgeBase.objects.filter(engine=self.object)
            .prefetch_related("owners")
            .annotate(
                cloudstorage_counter=Count("cloudstorage", distinct=True),
                localstorage_counter=Count("localstorage", distinct=True),
                website_counter=Count("website", distinct=True),
                urlbatch_counter=Count("urlbatch", distinct=True)
            ).order_by("created_at")
        )

        # Пагинация
        paginator = Paginator(base_qs, self.paginate_bases_by)
        page_number = self.request.GET.get("page")
        page_obj = paginator.get_page(page_number)

        context["base_list"] = page_obj
        context["page_obj"] = page_obj
        context["is_paginated"] = page_obj.has_other_pages()

        return context


class EngineCreateView(LoginRequiredMixin, PermissionRequiredMixin, CreateView):
    """Создание EmbeddingEngine с автоматическим указанием автора"""
    model = EmbeddingEngine
    fields = ["name", "model_name", "fine_tuning_params", "supports_multilingual", ]

    permission_required = "app_embeddings.add_embeddingengine"
    permission_denied_message = "У вас нет прав для создания объекта"
    template_name = "app_embeddings/engine_form.html"

    def get_form(self, form_class=None):
        form = super().get_form(form_class)
        form.fields["name"].widget.attrs.update({
            "class": "custom-field__input",
            "placeholder": " "
        })
        form.fields["model_name"].widget.attrs.update({
            "class": "custom-field__input",
            "placeholder": " "
        })
        form.fields["supports_multilingual"].widget.attrs.update({
            "class": "switch",
        })

        form.fields["fine_tuning_params"].widget = forms.Textarea(attrs={
            "class": "custom-field__input custom-field__input_wide custom-field__input_textarea",
            "placeholder": " "
        })
        return form

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)


class EngineUpdateView(LoginRequiredMixin, PermissionRequiredMixin, UpdateView):
    """Редактирование EmbeddingEngine"""
    model = EmbeddingEngine
    fields = ["name", "model_name", "fine_tuning_params", "supports_multilingual", ]

    permission_required = "app_embeddings.change_embeddingengine"
    permission_denied_message = "У вас нет прав для редактирования объекта"
    template_name = "app_embeddings/engine_form.html"

    def get_form(self, form_class=None):
        form = super().get_form(form_class)
        form.fields["name"].widget.attrs.update({
            "class": "custom-field__input",
            "placeholder": " "
        })
        form.fields["model_name"].widget.attrs.update({
            "class": "custom-field__input",
            "placeholder": " "
        })
        form.fields["supports_multilingual"].widget.attrs.update({
            "class": "switch",
        })

        form.fields["fine_tuning_params"].widget = forms.Textarea(attrs={
            "class": "custom-field__input custom-field__input_wide custom-field__input_textarea",
            "placeholder": " "
        })
        return form

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)


class EngineDeleteView(LoginRequiredMixin, PermissionRequiredMixin, DeleteView):
    """Удаление EmbeddingEngine"""
    model = EmbeddingEngine
    template_name = "app_embeddings/engine_confirm_delete.html"  # укажи свой шаблон
    success_url = reverse_lazy("embeddings:embeddingengine_list")  # замени на нужный URL
    permission_required = "app_embeddings.delete_embeddingengine"
    permission_denied_message = "У вас нет прав для удаления объекта"


class VectorizeWebsiteView(LoginRequiredMixin, StoragePermissionMixin, View):
    """
    Векторизация содержимого WebSite (создание эмбеддингов для чанков и сохранение в FAISS).
    """
    template_name = 'app_embeddings/vectorize_website.html'

    def get(self, request, pk):
        """Отображает форму подтверждения векторизации."""
        storage = get_object_or_404(WebSite, id=pk)
        kb = storage.kb
        context = {
            "kb": kb,
            "storage": storage,
            "storage_type_ru": storage._meta.verbose_name,
            "storage_type_eng": "website",
        }

        return render(request, self.template_name, context)

    def post(self, request, pk):
        # Получаем объект WebSite
        storage = get_object_or_404(WebSite, id=pk)
        kb = storage.kb
        engine = kb.engine

        context = {
            "kb": kb,
            "storage": storage,
            "storage_type_ru": storage._meta.verbose_name,
            "storage_type_eng": "website",
        }

        report_content = {
            "initial_data": {
                "engine": {
                    "model_name": engine.model_name,
                    "config": engine.fine_tuning_params,
                },
                # "objects": {
                #     "type": f"",
                #     "ids": [],
                # },
            }
        }

        embeddings_report = EmbeddingsReport.objects.create(
            site=storage,
            author=self.request.user,
            content=report_content,
        )
        try:
            task = create_vectors_task.delay(
                website_id=storage.pk,
                author_pk=request.user.pk,
                report_pk=embeddings_report.pk
            )

            embeddings_report.running_background_tasks[task.id] = "Векторизация чанков"
            embeddings_report.save(update_fields=["running_background_tasks", ])

            context = {
                "task_id": task.id,
                "task_name": f"Векторизация хранилища {storage._meta.verbose_name} {storage.name}",
                "task_object_url": storage.get_absolute_url(),  # TODO Поменять на вывод отчета
                "task_object_name": "Отчет о векторизации",
                "next_step_url": storage.get_absolute_url(),  # TODO Поменять на вывод отчета
            }
            return render(request=request,
                          template_name="celery_task_progress.html",
                          context=context
                          )

        except Exception as e:
            logger.exception(f"Ошибка векторизации хранилища: {storage.name}, запущена {request.user},"
                             f" отчет [EmbeddingsReport id {embeddings_report.pk}]")
            embeddings_report.content.setdefault("errors", []).append(str(e))
            embeddings_report.save()
            return render(request, 'app_sources/sync_result.html', {
                'result': {'error': f"Ошибка получения файлов: {e}"},
                'cloud_storage': cloud_storage
            })


class VectorizeStorageView(LoginRequiredMixin, StoragePermissionMixin, View):
    """
    Векторизация содержимого Storage (создание эмбеддингов для чанков и сохранение в FAISS).
    """
    template_name = 'app_embeddings/vectorize_website.html'

    STORAGES = {
        "cloud": CloudStorage,
        "local": LocalStorage,
    }

    def get(self, request, storage_type, pk):
        """Отображает форму подтверждения векторизации."""
        storage_cls = self.STORAGES.get(storage_type)
        if not storage_cls:
            raise ValueError("Неизвестный тип хранилища")
        storage = get_object_or_404(storage_cls, id=pk)
        kb = storage.kb
        context = {
            "kb": kb,
            "storage": storage,
            "storage_type_ru": storage._meta.verbose_name,
            "storage_type_eng": storage_type,
        }

        return render(request, self.template_name, context)

    def post(self, request, storage_type, pk):
        # Получаем объект WebSite
        storage_cls = self.STORAGES.get(storage_type)
        if not storage_cls:
            raise ValueError("Неизвестный тип хранилища")
        storage = get_object_or_404(storage_cls, id=pk)
        kb = storage.kb
        engine = kb.engine

        context = {
            "kb": kb,
            "storage": storage,
            "storage_type_ru": storage._meta.verbose_name,
            "storage_type_eng": "website",
        }

        report_content = {
            "initial_data": {
                "engine": {
                    "model_name": engine.model_name,
                    "config": engine.fine_tuning_params,
                },
                # "objects": {
                #     "type": f"",
                #     "ids": [],
                # },
            }
        }

        embeddings_report = EmbeddingsReport(
            author=self.request.user,
            content=report_content,
        )
        if isinstance(storage, CloudStorage):
            embeddings_report.cloud_storage = storage
        elif isinstance(storage, LocalStorage):
            embeddings_report.local_storage = storage
        elif isinstance(storage, WebSite):
            embeddings_report.site = storage
        elif isinstance(storage, URLBatch):
            embeddings_report.batch = storage
        embeddings_report.save()

        try:
            task = universal_create_vectors_task.delay(
                author_pk=request.user.pk,
                report_pk=embeddings_report.pk
            )

            embeddings_report.running_background_tasks[task.id] = "Векторизация чанков"
            embeddings_report.save(update_fields=["running_background_tasks", ])

            context = {
                "task_id": task.id,
                "task_name": f"Векторизация хранилища {storage._meta.verbose_name} {storage.name}",
                "task_object_url": storage.get_absolute_url(),  # TODO Поменять на вывод отчета
                "task_object_name": "Отчет о векторизации",
                "next_step_url": storage.get_absolute_url(),  # TODO Поменять на вывод отчета
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


class EngineTestTask(View):

    def get(self, request):
        task = test_task.delay(steps=60, sleep_per_step=60.0)
        context = {
                "task_id": task.id,
                "task_name": f"Тестовая задача",
                "task_object_url": reverse_lazy("embeddings:engine_list"),  # TODO Поменять на вывод отчета
                "task_object_name": "Отчет о векторизации",
                "next_step_url": reverse_lazy("embeddings:engine_list")  # TODO Поменять на вывод отчета
            }
        return render(request=request,
                      template_name="celery_task_progress.html",
                      context=context
                      )
