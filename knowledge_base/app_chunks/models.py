from enum import Enum

from django.contrib.auth import get_user_model
from django.db import models
from django.urls import reverse_lazy

from app_sources.content_models import URLContent
from app_sources.report_models import ChunkingReport

User = get_user_model()


class ChunkStatus(Enum):
    READY = "ready"
    ERROR = "error"
    CANCELED = "canceled"
    ACTIVE = "active"
    #
    # PARSED = "pa"
    # CHUNKED = "ch"
    # EMBEDDED = "em"
    # ADDED_TO_KB = "ad"

    @property
    def display_name(self):
        """Русское название статуса для отображения."""
        display_names = {
            "ready": "Готов к работе",
            "error": "Ошибка",
            "canceled": "Отменен",
            "active": "Активен",
            # "pa": "Обработан",
            # "ch": "Разбит на чанки",
            # "em": "Эмбеддинг выполнен",
            # "ad": "Добавлен в базу знаний",

        }
        return display_names.get(self.value, self.value)


class Chunk(models.Model):
    """
    Хранит чанки контента с адаптивными параметрами.
    """
    url_content = models.ForeignKey(URLContent, verbose_name="Разбиваемый URL контент", on_delete=models.CASCADE,
                                    blank=True, null=True)

    status = models.CharField(
        verbose_name="статус обработки",
        max_length=15,
        choices=[(status.value, status.display_name) for status in ChunkStatus],
        default=ChunkStatus.ERROR.value,
    )

    metadata = models.JSONField(verbose_name="metadata", default=dict, blank=True, null=True)
    page_content = models.TextField(help_text="контент чанка")
    splitter_cls = models.CharField(max_length=500, verbose_name="имя класса сплиттера")
    splitter_config = models.JSONField(verbose_name="конфигурация сплиттера", default=dict)

    report = models.ForeignKey(ChunkingReport, verbose_name="отчет", on_delete=models.CASCADE, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    author = models.ForeignKey(User, verbose_name="автор", on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Chunk"
        verbose_name_plural = "Chunks"

    def get_absolute_url(self):
        return reverse_lazy("chunks:chunk_detail", kwargs={"pk": self.pk})
