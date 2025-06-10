from django.contrib.auth import get_user_model
from django.db import models, transaction
from django.urls import reverse_lazy

from app_sources.storage_models import WebSite
from knowledge_base.base_models import TrackableModel
User = get_user_model()


class Parser(TrackableModel):
    """
    Абстрактный класс Парсера сайтов, хранящий информацию о примененном парсере BaseWebParser
    class_name - полный путь к модулю парсинга и config конфигурация парсера под конкретную задачу.
    """
    class_name = models.CharField(max_length=400, verbose_name="класс парсера из app_parsers.parsers.parser_classes")
    config = models.JSONField(verbose_name="конфигурация парсера", default=dict, blank=True)

    description = models.TextField(blank=True, verbose_name="Описание или цель использования")

    author = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="автор")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = models.Manager()

    class Meta:
        abstract = True
        # verbose_name = "Site Parser"
        # verbose_name_plural = "Site Parsers"


class MainParser(Parser):
    """Модель Основного парсера сайта, применяется для массового парсинга сайта"""
    site = models.OneToOneField(WebSite, on_delete=models.CASCADE, verbose_name="сайт")

    def get_absolute_url(self):
        return reverse_lazy("parsers:mainparser_detail", kwargs={"pk", self.pk})

    class Meta:
        verbose_name = "Парсер сайта"
        verbose_name_plural = "Парсеры сайтов"


class TestParser(Parser):
    """Модель тестового парсера применяется для теста отдельных страниц без изменения Основного парсера сайта"""
    site = models.ForeignKey(WebSite, on_delete=models.CASCADE, verbose_name="сайт", related_name="test_parsers")

    class Meta:
        verbose_name = "Тестовый парсер"
        verbose_name_plural = "Тестовые парсеры"
        constraints = [
            models.UniqueConstraint(fields=['site', 'author'], name='unique_test_parser_per_author_per_site')
        ]

    @classmethod
    @transaction.atomic
    def create_or_update(cls, *, site, author, **kwargs):
        """
        Создаёт или обновляет TestResult по уникальной паре (site, author).
        """
        obj, created = cls.objects.update_or_create(
            site=site,
            author=author,
            defaults=kwargs,
        )
        return obj, created


class TestParseReport(models.Model):
    """Класс результатов тестового запуска парсера для страницы сайта"""

    parser = models.OneToOneField(TestParser, on_delete=models.CASCADE, verbose_name="результат теста")

    url = models.URLField(verbose_name="тестовый url",)
    status = models.IntegerField(null=True, blank=True)
    html = models.TextField(null=True, blank=True)
    parsed_data = models.JSONField(null=True, blank=True)  # Для хранения результата парсинга
    error = models.TextField(null=True, blank=True)

    author = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="автор теста")
    created_at = models.DateTimeField(auto_now_add=True)
