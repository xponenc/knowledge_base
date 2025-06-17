from django import template

from app_sources.content_models import ContentStatus

register = template.Library()


@register.filter
def get_status_display(status_value):
    try:
        return ContentStatus(status_value).display_name
    except ValueError:

        return status_value or "—"
