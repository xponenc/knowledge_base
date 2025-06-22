from django import template

register = template.Library()


@register.filter
def get(dictionary, key):
    return dictionary.get(key)


@register.filter
def getlist(qs, key):
    return qs.getlist(key)
