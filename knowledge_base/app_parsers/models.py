from django.db import models


class Parser(models.Model):
    """
    Хранит информацию о парсерах для разных сайтов.
    """
    name = models.CharField(max_length=100, unique=True, help_text="Уникальное имя парсера")
    description = models.TextField(null=True, blank=True, help_text="Описание парсера")
    module_path = models.CharField(max_length=255, help_text="Путь к модулю парсера, например 'storages_external.site1_parser'")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Parser"
        verbose_name_plural = "Parsers"