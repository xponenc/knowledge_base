from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponseBadRequest, HttpResponse
from django.shortcuts import get_object_or_404, render, redirect
from django.template.loader import render_to_string
from django.views import View
from django.views.generic import DetailView

from app_parsers.forms import ParserDynamicConfigForm
from app_parsers.models import TestParser, MainParser
from app_parsers.services.parsers.dispatcher import WebParserDispatcher
from app_sources.storage_models import WebSite


class TestParserDetailView(LoginRequiredMixin, DetailView):
    """Детальный просмотр объекта класс :models:app_parsers.TestParser (Тестовый Парсер)"""
    model = TestParser


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


class MainParserDetailView(LoginRequiredMixin, DetailView):
    """Детальный просмотр объекта класс :models:app_parsers.MainParser (Основной Парсер)"""
    model = MainParser


class ParserSetTestAsMainView(LoginRequiredMixin, View):
    """Устанавливает тестовый парсер с конфигурацией как Основной парсер для сайта"""
    def get(self, request, pk, *args, **kwargs):
        test_parser = get_object_or_404(TestParser, pk=pk)
        website = test_parser.site
        main_parser = None
        if hasattr(website, "mainparser"):
            main_parser = website.mainparser
        context = {
            "site": website,
            "test_parser": test_parser,
            "main_parser": main_parser,

        }
        return render(request=request,
                      template_name="app_parsers/set_main_parser.html",
                      context=context)

    def post(self, request, pk, *args, **kwargs):
        test_parser = get_object_or_404(TestParser, pk=pk)
        website = test_parser.site
        main_parser = MainParser.objects.update_or_create(
            site=website,
            defaults={
                "class_name": test_parser.class_name,
                "config": test_parser.config,
                "description": test_parser.description,
                "author": request.user,
            }
        )

        return redirect(website.get_absolute_url())
