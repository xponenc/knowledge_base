from django import forms
from django.contrib import admin

from app_core.models import KnowledgeBase
from knowledge_base.admin_base import SoftDeleteAdmin


# class SoftDeleteAdminForm(forms.ModelForm):
#     hard_delete = forms.BooleanField(
#         label="Полное удаление",
#         required=False,
#         help_text="Отметьте для полного удаления (только для суперпользователя)."
#     )
#
#     class Meta:
#         model = KnowledgeBase
#         fields = '__all__'
#
#
# @admin.register(KnowledgeBase)
# class KnowledgeBaseAdmin(SoftDeleteAdmin):
#     form = SoftDeleteAdminForm
#
#     list_display = ('name', 'is_deleted', 'deleted_at')
#     list_filter = ('is_deleted',)
#     actions = ['hard_delete_selected']

@admin.register(KnowledgeBase)
class KnowledgeBaseAdmin(admin.ModelAdmin):
    search_fields = ['title']