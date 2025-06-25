from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.db.models import Func, IntegerField, F
from django.db.models.functions import Cast
from django.shortcuts import get_object_or_404, render, redirect
from django.views import View

from app_sources.source_models import URL, NetworkDocument, LocalDocument
from app_sources.storage_forms import StorageTagsForm, StorageScanTagsForm
from app_sources.storage_models import CloudStorage, LocalStorage, URLBatch, WebSite


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