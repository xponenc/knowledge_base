from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views import View
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.shortcuts import redirect, get_object_or_404, render
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

    def get_queryset(self):
        queryset = super().get_queryset()
        if self.request.user.is_superuser:
            return queryset
        queryset = queryset.filter(soft_deleted_at__isnull=True).filter(owners=self.request.user)
        return queryset


class KnowledgeBaseDetailView(LoginRequiredMixin, KBPermissionMixin, DetailView):
    """Детальная информация о базе знаний"""
    model = KnowledgeBase


class KnowledgeBaseCreateView(LoginRequiredMixin, CreateView):
    """Создание новой базы знаний"""
    model = KnowledgeBase
    fields = ['title', 'description', 'owners']
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
    fields = ['title', 'description', 'owners']
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
        return get_object_or_404(KnowledgeBase, pk=self.kwargs['pk'], soft_deleted_at__isnull=True)

    def get(self, request, *args, **kwargs):
        obj = self.get_object()
        return render(request, self.template_name, {'object': obj})

    def post(self, request, *args, **kwargs):
        obj = self.get_object()
        obj.soft_deleted_at = timezone.now()
        obj.log_history(request.user, 'delete', {})
        obj.save(update_fields=['soft_deleted_at', 'history'])
        return redirect(self.success_url)
