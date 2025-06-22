from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render
from django.views.generic import DetailView, UpdateView

from app_tasks.models import TaskForSource


class TaskForSourceDetailView(LoginRequiredMixin, DetailView):
    """Детальный просмотр задачи на изменение Источника данных"""
    # TODO права на просмотр только у владельцев Базы знаний
    model = TaskForSource


class TaskForSourceUpdateView(LoginRequiredMixin, UpdateView):
    """Детальный просмотр задачи на изменение Источника данных"""
    # TODO права на просмотр только у владельцев Базы знаний
    model = TaskForSource


    def get(self, request, *args, **kwargs):
        # TODO изменения возможны только для задач без установленного executor
        pass

    def post(self, request, *args, **kwargs):
        # TODO логика изменения статуса Источника в зависимости от статуса разрешения задачи
        # TODO проверка нескольких задач - если задач несколько то источник остается в статусе idle
        # TODO фиксировать завершение результатом и исполнителем
        pass