from django.contrib import admin
from .models import CustomerProfile


@admin.register(CustomerProfile)
class CustomerProfileAdmin(admin.ModelAdmin):
    list_display = (
        "id", "full_name", "telegram_id", "username",
        "role", "is_active", "knowledge_base", "created_at"
    )
    list_filter = ("role", "is_active", "created_at", "knowledge_base")
    search_fields = ("last_name", "first_name", "middle_name", "email", "telegram_id", "phone")
    readonly_fields = ("created_at",)
    autocomplete_fields = ("knowledge_base",)

    def full_name(self, obj):
        return f"{obj.last_name} {obj.first_name} {obj.middle_name or ''}".strip()
    full_name.short_description = "ФИО"

    def username(self, obj):
        return obj.user.username if obj.user else "—"

    username.short_description = "Аккаунт Django"

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related("knowledge_base")
