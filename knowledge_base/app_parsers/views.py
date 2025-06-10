from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponseBadRequest, HttpResponse
from django.shortcuts import get_object_or_404
from django.template.loader import render_to_string
from django.views import View

from app_parsers.forms import ParserDynamicConfigForm
from app_parsers.services.parsers.dispatcher import WebParserDispatcher
from app_sources.storage_models import WebSite


class ParserConfigView(LoginRequiredMixin, View):
    """Вью получения конфигурации Парсера"""
    def get(self, request, *args, **kwargs):
        parser_class_name = request.GET.get("parser_class_name")
        if not parser_class_name:
            return HttpResponseBadRequest("Параметр 'parser' обязателен")

        try:
            # parser_path может быть типа 'module.ClassName', нужно достать имя класса
            dispatcher = WebParserDispatcher()
            parser_cls = dispatcher.get_by_class_name(parser_class_name)
        except ValueError:
            return HttpResponseBadRequest("Парсер не найден")

        config_schema = getattr(parser_cls, "config_schema", {})
        config_form = ParserDynamicConfigForm(schema=config_schema)

        html = render_to_string("app_sources/include/partial_parser_config_form.html", {
            "config_form": config_form,
        }, request=request)

        return HttpResponse(html)


class ParserSetTestConfigAsMainView(LoginRequiredMixin, View):
    """Устанавливает тестовый парсер с конфигурацией как Основной парсер для сайта"""
    def post(self, request, website_pk, *args, **kwargs):
        website = get_object_or_404(WebSite, pk=website_pk)
        test

