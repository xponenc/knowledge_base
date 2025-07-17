from enum import Enum

from django.contrib.auth import get_user_model
from django.contrib.postgres.indexes import GinIndex
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import CheckConstraint, Q
from django.urls import reverse_lazy

from app_sources.content_models import URLContent, RawContent, CleanedContent
from app_sources.report_models import ReportStatus
from app_sources.storage_models import WebSite, URLBatch, CloudStorage, LocalStorage

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
                                    blank=True, null=True, related_name="chunks")
    raw_content = models.ForeignKey(RawContent, verbose_name="Разбиваемый исходный контент", on_delete=models.CASCADE,
                                    blank=True, null=True, related_name="chunks")
    cleaned_content = models.ForeignKey(CleanedContent, verbose_name="Разбиваемый чистый контент",
                                        on_delete=models.CASCADE,
                                        blank=True, null=True, related_name="chunks")

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

    report = models.ForeignKey("ChunkingReport", verbose_name="отчет", on_delete=models.CASCADE, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    author = models.ForeignKey(User, verbose_name="автор", on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Chunk"
        verbose_name_plural = "Chunks"

        indexes = [
            GinIndex(
                name="chunk_pgtrgm_idx",
                fields=["page_content"],
                opclasses=["gin_trgm_ops"],
            ),
        ]

    def get_absolute_url(self):
        return reverse_lazy("chunks:chunk_detail", kwargs={"pk": self.pk})


class ChunkingReport(models.Model):
    site = models.ForeignKey(WebSite, verbose_name="сайт", on_delete=models.CASCADE, blank=True, null=True)
    batch = models.ForeignKey(URLBatch, verbose_name="пакет", on_delete=models.CASCADE, blank=True, null=True)
    cloud_storage = models.ForeignKey(CloudStorage, verbose_name="облачное хранилище",
                                      on_delete=models.CASCADE, blank=True, null=True)
    local_storage = models.ForeignKey(LocalStorage, verbose_name="локальное хранилище",
                                      on_delete=models.CASCADE, blank=True, null=True)

    status = models.CharField(
        verbose_name="статус обработки",
        max_length=10,
        choices=[(status.value, status.display_name) for status in ReportStatus],
        default=ReportStatus.CREATED.value,
    )

    content = models.JSONField(verbose_name="отчет", default=dict)
    running_background_tasks = models.JSONField(verbose_name="выполняемые фоновые задачи по обработке отчета",
                                                default=dict)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    author = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        constraints = [
            CheckConstraint(
                check=(
                    (
                            Q(site__isnull=False, batch__isnull=True, cloud_storage__isnull=True,
                              local_storage__isnull=True) |
                            Q(site__isnull=True, batch__isnull=False, cloud_storage__isnull=True,
                              local_storage__isnull=True) |
                            Q(site__isnull=True, batch__isnull=True, cloud_storage__isnull=False,
                              local_storage__isnull=True) |
                            Q(site__isnull=True, batch__isnull=True, cloud_storage__isnull=True,
                              local_storage__isnull=False)
                    )
                ),
                name="only_one_source_must_be_set"
            )
        ]

    def clean(self):
        super().clean()

        fields = [self.site, self.batch, self.cloud_storage, self.local_storage]
        non_null_count = sum(1 for f in fields if f is not None)

        if non_null_count != 1:
            raise ValidationError(
                "Должно быть заполнено ровно одно из полей: site, batch, cloud_storage или local_storage")
