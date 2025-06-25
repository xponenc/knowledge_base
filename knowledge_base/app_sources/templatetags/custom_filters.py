from datetime import datetime

from django import template

register = template.Library()


@register.filter
def get(dictionary, key):
    return dictionary.get(key)


@register.filter
def getlist(qs, key):
    return qs.getlist(key)


@register.filter
def as_iso_date(value):
    try:
        # Пробуем DD.MM.YYYY
        if '.' in value:
            dt = datetime.strptime(value, "%d.%m.%Y")
        else:
            # Если уже ISO — пропускаем
            return value
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""
