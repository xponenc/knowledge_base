from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.db.models import Prefetch, Count
from django.views import View
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.shortcuts import redirect, get_object_or_404, render

from app_sources.storage_models import CloudStorage, WebSite, LocalStorage, URLBatch
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

    # def get_queryset(self):
    #     queryset = KnowledgeBase.objects.all(include_deleted=self.request.user.is_superuser)
    #     return queryset


class KnowledgeBaseDetailView(LoginRequiredMixin, KBPermissionMixin, DetailView):
    """Детальная информация о базе знаний"""
    model = KnowledgeBase
    cloud_storages = Prefetch(
        "cloudstorage_set",
        queryset=CloudStorage.objects.annotate(
            networkdocuments_counter=Count("network_documents", distinct=True)
        ),
        to_attr="cloud_storages"
    )
    local_storages = Prefetch(
        "cloudstorage_set",
        queryset=LocalStorage.objects.annotate(
            localdocuments_counter=Count("documents", distinct=True)
        ),
        to_attr="local_storages"
    )
    websites = Prefetch(
        "website_set",
        queryset=WebSite.objects.annotate(
            urls_counter=Count("url", distinct=True)
        ),
        to_attr="websites"
    )
    urlbatches = Prefetch(
        "website_set",
        queryset=URLBatch.objects.annotate(
            urls_counter=Count("url", distinct=True)
        ),
        to_attr="urlbatches"
    )

    queryset = KnowledgeBase.objects.prefetch_related(cloud_storages, local_storages, websites, urlbatches)


class KnowledgeBaseCreateView(LoginRequiredMixin, CreateView):
    """Создание новой базы знаний"""
    model = KnowledgeBase
    fields = ['name', 'description', 'owners']
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
    fields = ['name', 'description', 'owners']
    success_url = reverse_lazy('core:knowledgebase_list')

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
