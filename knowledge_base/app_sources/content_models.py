import os
from datetime import datetime
from enum import Enum
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import CheckConstraint, Q
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.urls import reverse_lazy
from django.utils.text import slugify

from app_sources.report_models import WebSiteUpdateReport, CloudStorageUpdateReport
from app_sources.source_models import NetworkDocument, LocalDocument, URL

User = get_user_model()


class ContentStatus(Enum):
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


class Content(models.Model):
    """Абстрактная модель объекта файла с данными из источника"""
    network_document = models.ForeignKey(NetworkDocument, on_delete=models.CASCADE, null=True, blank=True)
    local_document = models.ForeignKey(LocalDocument, on_delete=models.CASCADE, null=True, blank=True)
    status = models.CharField(
        verbose_name="статус обработки",
        max_length=15,
        choices=[(status.value, status.display_name) for status in ContentStatus],
        default=ContentStatus.READY.value,
    )

    report = models.ForeignKey(CloudStorageUpdateReport, verbose_name="создано в отчете", on_delete=models.CASCADE,
                               blank=True, null=True)

    hash_content = models.CharField(max_length=128, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    author = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)

    class Meta:
        abstract = True
        # Раскомментировать для PostgreSQL и убрать из clean
        constraints = [
            CheckConstraint(
                check=(
                    (
                            Q(network_document__isnull=False, local_document__isnull=True) |
                            Q(network_document__isnull=True, local_document__isnull=False)
                    )
                ),
                name='only_one_source_not_null'
            )
        ]

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
    # if instance.url:
    #     base_path = 'source_content/url_content'
    #     return f'{base_path}/url_{instance.url.id}_{timestamp}_raw_content.txt'
    if instance.local_document:
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

    def get_absolute_url(self):
        return reverse_lazy("sources:rawcontent_detail", kwargs={"pk": self.pk})


def get_cleaned_file_path(instance, filename):
    """Генерирует путь и имя файла для CleanedContent"""
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # if instance.url:
    #     base_path = 'source_content/url_content'
    #     return f'{base_path}/url_{instance.url.id}_{timestamp}_cleaned_content.txt'
    if instance.local_document or instance.network_document:
        document = instance.local_document if instance.local_document else instance.network_document
        base_path = 'source_content/document_content'
        prefix_name = "network_document" if instance.network_document else "local_document"
        return f'{base_path}/{prefix_name}_{document.id}_{timestamp}_cleaned_content.txt'
    return None


class CleanedContent(Content):
    """Файл с очищенным контентом источника"""
    raw_content = models.OneToOneField(RawContent, verbose_name="исходный документ", on_delete=models.CASCADE)
    file = models.FileField(verbose_name="файл с чистым контентом источника", upload_to=get_cleaned_file_path)
    preview = models.CharField(verbose_name="Превью", max_length=200, blank=True, null=True)
    recognition_method = models.CharField(verbose_name="метод распознавания контента", max_length=200,
                                          null=True, blank=True)
    recognition_quality = models.JSONField(verbose_name="отчет о качестве распознавания", default=dict)

    class Meta:
        verbose_name = "Cleaned Content"
        verbose_name_plural = "Cleaned Contents"

    @staticmethod
    def get_icon_class(self):
        return 'bi-filetype-txt'

    def get_absolute_url(self):
        return reverse_lazy("sources:cleanedcontent_detail", kwargs={"pk": self.pk})


@receiver(post_delete, sender=RawContent)
@receiver(post_delete, sender=CleanedContent)
def delete_content_file(sender, instance, **kwargs):
    """Удаляет файл при удалении записи RawContent или CleanedContent."""
    if instance.file and os.path.exists(instance.file.path):
        os.remove(instance.file.path)


class URLContent(Content):
    """Файл с очищенным контентом источника"""
    report = models.ForeignKey(WebSiteUpdateReport, verbose_name="создано в отчете", on_delete=models.CASCADE)

    url = models.ForeignKey(URL, on_delete=models.CASCADE)
    response_status = models.IntegerField(null=True, blank=True, help_text="HTTP-код ответа")

    body = models.TextField(verbose_name="полезный контент", null=True, blank=True)
    metadata = models.JSONField(verbose_name="метаданные", null=True, blank=True)
    # hash = models.CharField(verbose_name="хеш контента для анализа изменений", max_length=128, null=True, blank=True)
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

    # created_at = models.DateTimeField(auto_now_add=True)
    # updated_at = models.DateTimeField(auto_now=True)
    #
    # author = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)

    class Meta:
        verbose_name = "URL Cleaned Content"
        verbose_name_plural = "URL Cleaned Contents"

    def get_absolute_url(self):
        return reverse_lazy("sources:urlcontent_detail", kwargs={"pk": self.pk})
