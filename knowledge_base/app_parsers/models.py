from django.db import models


class Parser(models.Model):
    """
    Хранит информацию о парсерах c конфигурацией парсера для разных сайтов.
    """
    class_name = models.CharField(max_length=400, verbose_name="класс парсера из app_parsers.parsers.parser_classes")
    config = models.JSONField(verbose_name="конфигурация парсера", default=dict, blank=True)

    description = models.TextField(blank=True, verbose_name="Описание или цель использования")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Site Parser"
        verbose_name_plural = "Site Parsers"
    #
    # def get_parser_instance(self) -> BaseWebParser:
    #     cls = WebParserDispatcher().get_by_class_name(self.class_name)
    #     return cls(config=self.config)
