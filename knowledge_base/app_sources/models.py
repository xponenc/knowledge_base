import os
from datetime import datetime
from importlib import import_module
from pathlib import Path

from django.core.exceptions import ValidationError
from django.db import models
from django.contrib.postgres.indexes import GinIndex
from enum import Enum
import hashlib

from django.db.models import UniqueConstraint, Q, CheckConstraint
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.urls import reverse
from django.utils.text import slugify
from app_sources.storage_models import CloudStorage, CloudStorageUpdateReport, LocalStorage
from django.contrib.auth import get_user_model

User = get_user_model()


class Status(Enum):
    CREATED = "cr"
    SYNCED = "sy"
    PARSED = "pa"
    CHUNKED = "ch"
    EMBEDDED = "em"
    ADDED_TO_KB = "ad"
    DELETED = "de"
    EXCLUDED = "ex"
    ERROR = "er"

    @property
    def display_name(self):
        """Русское название статуса для отображения."""
        display_names = {
            "cr": "Создан",
            "sy": "Синхронизирован",
            "pa": "Обработан",
            "ch": "Разбит на чанки",
            "em": "Эмбеддинг выполнен",
            "ad": "Добавлен в базу знаний",
            "de": "Удален",
            "ex": "Исключен из базы знаний",
            "er": "Ошибка"
        }
        return display_names.get(self.value, self.value)


class AbstractSource(models.Model):
    """
    Абстрактная базовая модель для источников данных (URL, Document).
    """
    url = models.URLField(verbose_name="путь к источнику", max_length=500)

    status = models.CharField(
        verbose_name="статус обработки",
        max_length=2,
        choices=[(status.value, status.display_name) for status in Status],
        default=Status.CREATED.value,
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
    categories = models.JSONField(verbose_name="список тегов",
                                  default=list,
                                  help_text="Категории в формате JSON, например ['news', 'tech', ]")
    error_message = models.CharField(verbose_name="ошибка при обрабоотке",
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

    response_status = models.IntegerField(null=True, blank=True, help_text="HTTP-код ответа")

    def __str__(self):
        return f"[URL] {super().__str__()}"

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
            "t": "Документ передается в базу знаний в виде текста",
            "f": "Документ передается в базу знаний в виде файла",
        }
        return display_names.get(self.value, self.value)


class NetworkDocument(AbstractSource):
    """Документ связанный с сетевым хранилищем"""

    storage = models.ForeignKey(CloudStorage, on_delete=models.CASCADE, related_name="network_documents")

    path = models.URLField(verbose_name="путь к источнику", max_length=500)
    file_id = models.CharField(verbose_name="id файла в облаке", max_length=200, blank=True, null=True)
    size = models.PositiveBigIntegerField(verbose_name="размер файла")
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
        # Раскомментировать для PostgreSQL и убрать из clean
        # constraints = [
        #     UniqueConstraint(
        #         fields=['storage', 'path'],
        #         condition=Q(storage__isnull=False, path__isnull=False),
        #         name='unique_cloudstorage_path_not_null'
        #     )
        # ]

    def __str__(self):
        return f"[NetworkDocument] {super().__str__()}"

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

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    author = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Local Document"
        verbose_name_plural = "Local Documents"

    def __str__(self):
        return f"[LocalDocument] {super().__str__()}"







class Content(models.Model):
    """Абстрактная модель объекта файла с данными из источника"""
    url = models.ForeignKey(URL, on_delete=models.CASCADE, null=True, blank=True)
    network_document = models.ForeignKey(NetworkDocument, on_delete=models.CASCADE, null=True, blank=True)
    local_document = models.ForeignKey(LocalDocument, on_delete=models.CASCADE, null=True, blank=True)

    hash_content = models.CharField(max_length=64, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    author = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        abstract = True
        # Раскомментировать для PostgreSQL и убрать из clean
        # constraints = [
        #     CheckConstraint(
        #         check=(
        #             (
        #                     Q(url__isnull=False, network_document__isnull=True, local_document__isnull=True) |
        #                     Q(url__isnull=True, network_document__isnull=False, local_document__isnull=True) |
        #                     Q(url__isnull=True, network_document__isnull=True, local_document__isnull=False)
        #             )
        #         ),
        #         name='only_one_source_not_null'
        #     )
        # ]

    objects = models.Manager()

    def clean(self):
        """Ровно одно из полей должно быть заполнено."""
        sources = [self.url, self.network_document, self.local_document]
        if sum(bool(s) for s in sources) != 1:
            raise ValidationError(
                "Должно быть заполнено ровно одно из полей: url, network_document, local_document.")


def get_raw_file_path(instance, filename):
    """Генерирует путь и имя файла для RawContent"""
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if instance.url:
        base_path = 'source_content/url_content'
        return f'{base_path}/url_{instance.url.id}_{timestamp}_raw_content.txt'
    elif instance.local_document:
        base_path = 'source_content/document_content'
        original_filename = filename or 'document'
        sanitized_filename = slugify(os.path.splitext(original_filename)[0]) + os.path.splitext(original_filename)[1]
        return f'{base_path}/local_document_{instance.local_document.pk}_{timestamp}_{sanitized_filename}'
    elif instance.network_document:
        base_path = 'source_content/document_content'
        original_filename = filename or 'document'
        sanitized_filename = slugify(os.path.splitext(original_filename)[0]) + os.path.splitext(original_filename)[1]
        return f'{base_path}/network_document_{instance.network_document.pk}_{timestamp}_{sanitized_filename}'
    return None


class RawContent(Content):
    """Файл с грязным контентом источника"""
    file = models.FileField(verbose_name="файл с грязным контентом источника", upload_to=get_raw_file_path)

    class Meta:
        verbose_name = "Raw Content"
        verbose_name_plural = "Raw Contents"

    def file_extension(self):
        """Возвращает расширение файла (в нижнем регистре)"""
        if self.file:
            return Path(self.file.name).suffix.lower()
        return ''

    def is_image(self):
        """Проверяет, является ли файл изображением"""
        return self.file_extension() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']

    def get_icon_class(self):
        ext = self.file_extension()
        icon_map = {
            '.pdf': 'bi-filetype-pdf',
            '.doc': 'bi-filetype-doc',
            '.docx': 'bi-filetype-docx',
            '.xls': 'bi-filetype-xls',
            '.xlsx': 'bi-filetype-xlsx',
            '.txt': 'bi-filetype-txt',
            '.csv': 'bi-filetype-csv',
            '.zip': 'bi-file-earmark-zip',
        }
        return icon_map.get(ext, 'bi-file-earmark')


def get_cleaned_file_path(instance, filename):
    """Генерирует путь и имя файла для CleanedContent"""
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if instance.url:
        base_path = 'source_content/url_content'
        return f'{base_path}/url_{instance.url.id}_{timestamp}_cleaned_content.txt'
    elif instance.local_document or instance.network_document:
        document = instance.local_document if instance.local_document else instance.network_document
        base_path = 'source_content/document_content'
        return f'{base_path}/document_{document.id}_{timestamp}_cleaned_content.txt'
    return None


class CleanedContent(Content):
    """Файл с очищенным контентом источника"""
    raw_content = models.OneToOneField(RawContent, verbose_name="исходный документ", on_delete=models.CASCADE)
    file = models.FileField(verbose_name="файл с чистым контентом источника", upload_to=get_cleaned_file_path)

    class Meta:
        verbose_name = "Cleaned Content"
        verbose_name_plural = "Cleaned Contents"

    @staticmethod
    def get_icon_class(self):
        return 'bi-filetype-txt'


@receiver(post_delete, sender=RawContent)
@receiver(post_delete, sender=CleanedContent)
def delete_content_file(sender, instance, **kwargs):
    """Удаляет файл при удалении записи RawContent или CleanedContent."""
    if instance.file and os.path.exists(instance.file.path):
        os.remove(instance.file.path)
