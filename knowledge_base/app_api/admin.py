from django.contrib import admin
from .models import ApiClient


@admin.register(ApiClient)
class ApiClientAdmin(admin.ModelAdmin):
    list_display = ('name', 'knowledge_base', 'is_active', 'created_at')
    list_filter = ('is_active', 'knowledge_base')
    search_fields = ('name', 'token')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('knowledge_base')