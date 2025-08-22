import json
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def pretty_dict(value, indent=2):
    """
    Форматирует dict/list/tuple в красивый JSON с отступами.
    Если пришла строка — попробует распарсить JSON, иначе вернёт как есть.
    """
    try:
        if isinstance(value, (dict, list, tuple)):
            return mark_safe(json.dumps(value, ensure_ascii=False, indent=int(indent)))
        # строка: попробуем как JSON
        try:
            obj = json.loads(value)
            return mark_safe(json.dumps(obj, ensure_ascii=False, indent=int(indent)))
        except Exception:
            return value
    except Exception:
        return value