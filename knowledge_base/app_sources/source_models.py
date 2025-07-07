from enum import Enum

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import UniqueConstraint, Q
from django.urls import reverse_lazy

from app_sources.report_models import WebSiteUpdateReport, CloudStorageUpdateReport
from app_sources.storage_models import WebSite, URLBatch, CloudStorage, LocalStorage
from knowledge_base.mixin_models import TrackableModel, SoftDeleteModel

User = get_user_model()


class SourceStatus(Enum):
    CREATED = "cr"
    READY = "ready"
    DELETED = "de"
    EXCLUDED = "ex"
    ERROR = "er"
    WAIT = "wait"

    @property
    def display_name(self):
        """Русское название статуса для отображения."""
        display_names = {
            "cr": "Создан",
            "ready": "Готов",
            "de": "Удален",
            "ex": "Исключен из базы знаний",
            "er": "Ошибка",
            "wait": "ожидает решения задачи",
        }
        return display_names.get(self.value, self.value)


class AbstractSource(TrackableModel):
    """
    Абстрактная базовая модель для источников данных (URL, Document).
    """
    url = models.URLField(verbose_name="путь к источнику", max_length=500)

    status = models.CharField(
        verbose_name="статус обработки",
        max_length=10,
        choices=[(status.value, status.display_name) for status in SourceStatus],
        default=SourceStatus.CREATED.value,
    )
    language = models.CharField(verbose_name="язык документа",
                                max_length=10,
                                null=True,
                                blank=True,
                                )
    title = models.CharField(verbose_name="наименование документа",
                             max_length=500,
                             blank=True,
                             null=True)
    tags = models.JSONField(verbose_name="список тегов",
                            default=list,
                            blank=True,
                            help_text="Категории в формате JSON, например ['news', 'tech', ]")
    error_message = models.CharField(verbose_name="ошибка при обработке",
                                     max_length=1000,
                                     null=True,
                                     blank=True,
                                     )
    metadata = models.JSONField(
        verbose_name="дополнительные метаданные",
        default=dict,
        help_text="Хранит API-специфичные данные (например, fileId для Google Drive)"
    )

    remote_updated = models.DateTimeField(verbose_name="дата обновления в источнике",
                                          null=True,
                                          blank=True,
                                          )
    local_updated = models.DateTimeField(verbose_name="дата обновления в проекте",
                                         null=True,
                                         blank=True,
                                         )
    synchronized_at = models.DateTimeField(verbose_name="дата синхронизации",
                                           null=True,
                                           blank=True,
                                           )
    created_at = models.DateTimeField(verbose_name="дата создания",
                                      auto_now_add=True)
    updated_at = models.DateTimeField(verbose_name="дата обновления", auto_now=True)
    soft_deleted_at = models.DateTimeField(verbose_name="удален из базы знаний",
                                           blank=True,
                                           null=True)

    class Meta:
        abstract = True

    objects = models.Manager()

    def __str__(self):
        return self.title if self.title else self.url


class URL(AbstractSource):
    """Модель страницы сайта с информацией"""
    site = models.ForeignKey(WebSite, verbose_name="сайт", on_delete=models.CASCADE, blank=True, null=True)
    batch = models.ForeignKey(URLBatch, verbose_name="пакет", on_delete=models.CASCADE, blank=True, null=True)
    report = models.ForeignKey(WebSiteUpdateReport, verbose_name="создано в отчете", on_delete=models.CASCADE,
                               blank=True, null=True)

    def __str__(self):
        return f"[URL] {super().__str__()}"

    def get_absolute_url(self):
        return reverse_lazy("sources:url_detail", kwargs={"pk": self.pk, })

    class Meta:
        verbose_name = "URL"
        verbose_name_plural = "URLs"


class OutputDataType(Enum):
    """Тип передачи документа в базу знаний"""
    text = "t"
    file = "f"

    @property
    def display_name(self):
        """Русское название статуса для отображения."""
        display_names = {
            "t": "текст",
            "f": "файл",
        }
        return display_names.get(self.value, self.value)


class NetworkDocument(AbstractSource):
    """Документ связанный с сетевым хранилищем"""

    storage = models.ForeignKey(CloudStorage, on_delete=models.CASCADE, related_name="documents")
    report = models.ForeignKey(CloudStorageUpdateReport, verbose_name="создано в отчете", on_delete=models.CASCADE, blank=True, null=True)
    path = models.URLField(verbose_name="путь к источнику", max_length=500)
    file_id = models.CharField(verbose_name="id файла в облаке", max_length=200, blank=True, null=True)
    output_format = models.CharField(
        verbose_name="формат вывода документа в базу знаний",
        max_length=1,
        choices=[(status.value, status.display_name) for status in OutputDataType],
        default=OutputDataType.file.value,
    )
    description = models.CharField(verbose_name="краткое описание", max_length=1000, null=True, blank=True)

    class Meta:
        verbose_name = "Network Document"
        verbose_name_plural = "Network Documents"
        constraints = [
            UniqueConstraint(
                fields=['storage', 'path'],
                condition=Q(storage__isnull=False, path__isnull=False),
                name='unique_cloudstorage_path_not_null'
            )
        ]

    def __str__(self):
        return f"[NetworkDocument] {super().__str__()}"

    def get_absolute_url(self):
        return reverse_lazy("sources:networkdocument_detail", kwargs={"pk": self.pk, })

    def clean(self):
        # Только если оба поля не None — проверка на уникальность
        if self.storage and self.url:
            if NetworkDocument.objects.filter(storage=self.storage, url=self.url).exclude(pk=self.pk).exists():
                raise ValidationError("Документ с таким storage и url уже существует.")


class LocalDocument(AbstractSource):
    """Модель документа привязанного к локальному хранилищу документов"""

    storage = models.ForeignKey(LocalStorage, on_delete=models.CASCADE, related_name="documents")

    name = models.CharField(max_length=200, unique=True, help_text="название")
    description = models.CharField(verbose_name="описание", max_length=1000, null=True, blank=True)
    output_format = models.CharField(
        verbose_name="формат вывода документа в базу знаний",
        max_length=1,
        choices=[(status.value, status.display_name) for status in OutputDataType],
        default=OutputDataType.file.value,
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    author = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Local Document"
        verbose_name_plural = "Local Documents"

    def __str__(self):
        return f"[LocalDocument] {super().__str__()}"

    def get_absolute_url(self):
        return reverse_lazy("sources:localdocument_detail", kwargs={"pk": self.pk, })