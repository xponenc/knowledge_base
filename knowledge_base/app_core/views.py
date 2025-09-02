from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.db.models import Prefetch, Count, Subquery, OuterRef, IntegerField, Q, F
from django.db.models.functions import Coalesce
from django.views import View
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.shortcuts import redirect, get_object_or_404, render

from app_embeddings.models import Embedding
from app_sources.source_models import NetworkDocument, LocalDocument, URL
from app_sources.storage_models import CloudStorage, WebSite, LocalStorage, URLBatch
from .forms import KnowledgeBaseForm
from .models import KnowledgeBase
from django.utils import timezone


class KBPermissionMixin(UserPassesTestMixin):
    """Mixin: Проверяет, что пользователь — владелец или суперюзер"""
    def test_func(self):
        obj = self.get_object()
        return obj.is_owner_or_superuser(self.request.user)


class KnowledgeBaseListView(LoginRequiredMixin, ListView):
    """Список баз знаний (только доступные пользователю)"""
    model = KnowledgeBase
    paginate_by = 3

    def get_queryset(self):
        """Возвращает отфильтрованный и аннотированный queryset"""
        user = self.request.user
        qs = KnowledgeBase.objects.prefetch_related("owners").annotate(
            cloudstorage_counter=Count("cloudstorage", distinct=True),
            localstorage_counter=Count("localstorage", distinct=True),
            website_counter=Count("website", distinct=True),
            urlbatch_counter=Count("urlbatch", distinct=True),
        ).order_by("created_at")

        if not user.is_superuser:
            qs = qs.filter(owners=user)
        return qs

    def get(self, request, *args, **kwargs):
        """Обрабатывает GET-запрос с учетом редиректа для пользователей с одной базой знаний"""
        user = request.user
        if not user.is_superuser:

            owned = self.get_queryset()
            if owned.count() == 1:
                return redirect(owned.first().get_absolute_url())
        return super().get(request, *args, **kwargs)

    def get_context_data(self, *args, **kwargs):
        """Добавляет количество баз знаний в контекст"""
        context = super().get_context_data(*args, **kwargs)
        context["kb_counter"] = len(context["object_list"]) if context["object_list"] else self.get_queryset().count()
        return context


class KnowledgeBaseDetailView(LoginRequiredMixin, KBPermissionMixin, DetailView):
    """Детальная информация о базе знаний"""
    model = KnowledgeBase

    def get_queryset(self):
        cloud_docs_subquery = Subquery(
            NetworkDocument.objects
            .filter(storage=OuterRef("pk"))
            .order_by()
            .values("storage")
            .annotate(cnt=Count("id"))
            .values("cnt")[:1],
            output_field=IntegerField()
        )
        cloud_embeddings_subquery = Subquery(
            Embedding.objects
            .filter(
                Q(chunk__raw_content__network_document__storage=OuterRef("pk")) |
                Q(chunk__cleaned_content__network_document__storage=OuterRef("pk"))
            )
            .annotate(storage_id=Coalesce(
                F("chunk__raw_content__network_document__storage"),
                F("chunk__cleaned_content__network_document__storage"),
            ))
            .values("storage_id")  # группировка по вычисленному storage
            .annotate(cnt=Count("id"))
            .values("cnt")[:1],
            output_field=IntegerField()
        )

        cloud_storages = Prefetch(
            "cloudstorage_set",
            queryset=CloudStorage.objects
            .select_related("author", )
            .annotate(
                networkdocuments_counter=cloud_docs_subquery,
                embeddings_counter=cloud_embeddings_subquery,
            ),
            to_attr="cloud_storages"
        )


        # Local storage
        local_docs_subquery = Subquery(
            LocalDocument.objects
            .filter(storage=OuterRef("pk"))
            .order_by()
            .values("storage")
            .annotate(cnt=Count("id"))
            .values("cnt")[:1],
            output_field=IntegerField()
        )
        local_embeddings_subquery = Subquery(
            Embedding.objects
            .filter(
                Q(chunk__raw_content__local_document__storage=OuterRef("pk")) |
                Q(chunk__cleaned_content__local_document__storage=OuterRef("pk"))
            )
            .annotate(storage_id=Coalesce(
                F("chunk__raw_content__local_document__storage"),
                F("chunk__cleaned_content__local_document__storage"),
            ))
            .values("storage_id")  # группировка по вычисленному storage
            .annotate(cnt=Count("id"))
            .values("cnt")[:1],
            output_field=IntegerField()
        )

        local_storages = Prefetch(
            "localstorage_set",
            queryset=LocalStorage.objects
            .select_related("author")
            .annotate(
                localdocuments_counter=local_docs_subquery,
                embeddings_counter=cloud_embeddings_subquery,
            ),
            to_attr="local_storages"
        )

        # Websites
        site_urls_subquery = Subquery(
            URL.objects
            .filter(site=OuterRef("pk"))
            .order_by()
            .values("site")
            .annotate(cnt=Count("id"))
            .values("cnt")[:1],
            output_field=IntegerField()
        )

        site_embeddings_subquery = Subquery(
            Embedding.objects
            .filter(chunk__url_content__url__site=OuterRef("pk"))
            .values("chunk__url_content__url__site")  # группировка по сайту
            .annotate(cnt=Count("id"))
            .values("cnt")[:1],
            output_field=IntegerField()
        )

        websites = Prefetch(
            "website_set",
            queryset=WebSite.objects
            .select_related("author")
            .annotate(
                urls_counter=site_urls_subquery,
                embeddings_counter=site_embeddings_subquery,
            ),
            to_attr="websites"
        )

        # URLBatch
        batch_urls_subquery = Subquery(
            URL.objects
            .filter(batch=OuterRef("pk"))
            .order_by()
            .values("batch")
            .annotate(cnt=Count("id"))
            .values("cnt")[:1],
            output_field=IntegerField()
        )

        batch_embeddings_subquery = Subquery(
            Embedding.objects
            .filter(chunk__url_content__url__batch=OuterRef("pk"))
            .values("chunk__url_content__url__site")  # группировка по сайту
            .annotate(cnt=Count("id"))
            .values("cnt")[:1],
            output_field=IntegerField()
        )

        urlbatches = Prefetch(
            "urlbatch_set",
            queryset=URLBatch.objects
            .select_related("author")
            .annotate(
                urls_counter=batch_urls_subquery,
                embeddings_counter=batch_embeddings_subquery,

            ),
            to_attr="urlbatches"
        )

        queryset = (
            KnowledgeBase.objects
            .select_related("engine")
            .prefetch_related(
                cloud_storages,
                local_storages,
                websites,
                urlbatches,
            )
        )
        return queryset


class KnowledgeBaseCreateView(LoginRequiredMixin, CreateView):
    """Создание новой базы знаний"""
    model = KnowledgeBase
    form_class = KnowledgeBaseForm
    success_url = reverse_lazy('core:knowledgebase_list')

    def form_valid(self, form):
        response = super().form_valid(form)
        if self.request.user not in form.instance.owners.all():
            form.instance.owners.add(self.request.user)
        form.instance.log_history(self.request.user, 'create', {})
        form.instance.save()
        return response


class KnowledgeBaseUpdateView(LoginRequiredMixin, KBPermissionMixin, UpdateView):
    """Редактирование базы знаний с логированием изменений"""
    model = KnowledgeBase
    form_class = KnowledgeBaseForm
    # success_url = reverse_lazy('core:knowledgebase_list')

    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.save_with_log(self.request.user)
        form.save_m2m()
        return redirect(self.get_success_url())


class KnowledgeBaseDeleteView(LoginRequiredMixin, KBPermissionMixin, View):
    """Мягкое удаление базы знаний с логированием"""
    template_name = 'app_core/knowledgebase_confirm_delete.html'
    success_url = reverse_lazy('core:knowledgebase_list')

    def get_object(self):
        return get_object_or_404(KnowledgeBase, pk=self.kwargs['pk'])

    def get(self, request, *args, **kwargs):
        obj = self.get_object()
        return render(request, self.template_name, {'object': obj})

    def post(self, request, *args, **kwargs):
        obj = self.get_object()
        obj.delete()
        obj.log_history(request.user, 'delete', {})
        obj.save()
        return redirect(self.success_url)
