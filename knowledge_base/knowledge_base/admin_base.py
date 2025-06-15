# core/admin/base.py

from django.contrib import admin
from django.contrib.admin import ModelAdmin


class SoftDeleteAdmin(ModelAdmin):
    def get_queryset(self, request):
        """Показывать удаленные объекты для суперпользователя."""
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs.all(include_deleted=True)
        return qs

    def delete_model(self, request, obj):
        """Переопределяем удаление одного объекта."""
        if request.user.is_superuser and 'hard_delete' in request.POST:
            obj.delete(hard_delete=True)
        else:
            obj.delete()

    def delete_queryset(self, request, queryset):
        """Переопределяем массовое удаление."""
        if request.user.is_superuser and 'hard_delete' in request.POST:
            queryset.hard_delete()
        else:
            queryset.delete()

    def get_actions(self, request):
        """Добавляем действие полного удаления для суперпользователя."""
        actions = super().get_actions(request)
        if request.user.is_superuser:
            def hard_delete_selected(modeladmin, request, queryset):
                queryset.hard_delete()
            hard_delete_selected.short_description = "Полное удаление выбранных объектов"
            actions['hard_delete_selected'] = hard_delete_selected
        return actions
