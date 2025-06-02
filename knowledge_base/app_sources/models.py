import os
from datetime import datetime
from importlib import import_module

from django.core.exceptions import ValidationError
from django.db import models
from django.contrib.postgres.indexes import GinIndex
from enum import Enum
import hashlib

from django.db.models import UniqueConstraint, Q
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.urls import reverse
from django.utils.text import slugify


class Status(Enum):
    CREATED = "cr"
    SYNCED = "sy"
    PARSED = "pa"
    CHUNKED = "ch"
    EMBEDDED = "em"
    ADDED_TO_KB = "ad"
    DELETED = "de"
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

    def __str__(self):
        return self.title if self.title else self.path


class URL(AbstractSource):
    """
    Хранит данные о веб-страницах, включая категории и язык.
    """
    # parser = models.ForeignKey(Parser, on_delete=models.SET_NULL, null=True, related_name="urls",
    #                            help_text="Связанный парсер")
    response_status = models.IntegerField(null=True, blank=True, help_text="HTTP-код ответа")

    def __str__(self):
        return f"[URL] {super().__str__()}"

    class Meta:
        verbose_name = "URL"
        verbose_name_plural = "URLs"


class CloudStorage(models.Model):
    """
    Хранит информацию о подключенных облачных дисках.
    """

    # Диспетчер классов хранилищ
    STORAGE_CLASSES = {
        'webdav': 'storages_external.webdav_storage.webdav_client.WebDavStorage',
        # 'google_drive': 'storages_external.google_drive.google_drive_client.GoogleDriveStorage',
        # 'dropbox': 'storages_external.dropbox.dropbox_client.DropboxStorage',
    }

    name = models.CharField(max_length=100, unique=True, help_text="Название диска")
    api_type = models.CharField(max_length=100, choices=((key, key) for key in STORAGE_CLASSES))
    credentials = models.JSONField(null=True, blank=True, help_text="Учетные данные")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Cloud Storage"
        verbose_name_plural = "Cloud Storages"

    objects = models.Manager()

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse("sources:cloud_storage_detail", kwargs={"pk": self.id, })

    def get_storage(self):
        """
        Возвращает экземпляр класса хранилища на основе api_type.

        Args:
            cloud_storage: Объект CloudStorage.

        Returns:
            Экземпляр класса хранилища (например, WebDavStorage).

        Raises:
            ValueError: Если api_type не поддерживается.
        """
        if self.api_type in self.STORAGE_CLASSES:
            module_path, class_name = self.STORAGE_CLASSES[self.api_type].rsplit('.', 1)
            module = import_module(module_path)
            storage_class = getattr(module, class_name)
            return storage_class(self.credentials)
        raise ValueError(f"Неподдерживаемый api_type: {self.api_type}")


class StorageUpdateReport(models.Model):
    """Отчет по обновлению файлов Облачного хранилища"""

    storage = models.ForeignKey(CloudStorage, on_delete=models.CASCADE, )
    content = models.JSONField(verbose_name="отчет", default=dict)
    created_at = models.DateTimeField(auto_now_add=True)


class DocumentSourceType(Enum):
    """Тип размещения источника исходного документа"""
    network = "n"
    local = "l"

    @property
    def display_name(self):
        """Русское название статуса для отображения."""
        display_names = {
            "n": "Источник документа - сетевой диск",
            "l": "Источник документа - локальный файл",
        }
        return display_names.get(self.value, self.value)


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


class Document(AbstractSource):
    """
    Хранит данные о документах с облачных дисков.
    """
    cloud_storage = models.ForeignKey(CloudStorage, on_delete=models.CASCADE, related_name="documents")
    path = models.URLField(verbose_name="путь к источнику", max_length=500)
    file_id = models.CharField(verbose_name="id файла в облаке", max_length=200, blank=True, null=True)
    size = models.PositiveBigIntegerField(verbose_name="размер файла")
    output_format = models.CharField(
        verbose_name="формат вывода документа в базу знаний",
        max_length=1,
        choices=[(status.value, status.display_name) for status in OutputDataType],
        default=OutputDataType.file.value,
    )
    source_type = models.CharField(
        verbose_name="тип исходного источника документа",
        max_length=1,
        choices=[(status.value, status.display_name) for status in DocumentSourceType],
        default=DocumentSourceType.network.value,
    )

    description = models.TextField(null=True, blank=True, help_text="Краткое описание")

    def __str__(self):
        return f"[Document] {super().__str__()}"

    def clean(self):
        # Только если оба поля не None — проверка на уникальность
        if self.cloud_storage and self.url:
            if Document.objects.filter(cloud_storage=self.cloud_storage, url=self.url).exclude(pk=self.pk).exists():
                raise ValidationError("Документ с таким cloud_storage и url уже существует.")

    class Meta:
        verbose_name = "Document"
        verbose_name_plural = "Documents"
        # Раскомментировать для PostgreSQL и убрать из clean
        # constraints = [
        #     UniqueConstraint(
        #         fields=['cloud_storage', 'path'],
        #         condition=Q(cloud_storage__isnull=False, path__isnull=False),
        #         name='unique_cloudstorage_path_not_null'
        #     )
        # ]


def get_raw_file_path(instance, filename):
    """Генерирует путь и имя файла для RawContent и CleanedContent."""
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if instance.url:
        base_path = 'source_content/url_content'
        return f'{base_path}/url_{instance.url.id}_{timestamp}_raw_content.txt'
    elif instance.document:
        base_path = 'source_content/document_content'
        original_filename = filename or 'document'
        sanitized_filename = slugify(os.path.splitext(original_filename)[0]) + os.path.splitext(original_filename)[1]
        print(f'{base_path}/document_{instance.document.id}_{timestamp}_{sanitized_filename}')
        return f'{base_path}/document_{instance.document.id}_{timestamp}_{sanitized_filename}'
    return None


def get_cleaned_file_path(instance, filename):
    """Генерирует путь и имя файла для RawContent и CleanedContent."""
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if instance.url:
        base_path = 'source_content/url_content'
        return f'{base_path}/url_{instance.url.id}_{timestamp}_cleaned_content.txt'
    elif instance.document:
        base_path = 'source_content/document_content'
        return f'{base_path}/document_{instance.document.id}_{timestamp}_cleaned_content.txt'
    return None


class RawContent(models.Model):
    """Файл с грязным контентом источника"""
    url = models.ForeignKey(URL, on_delete=models.CASCADE, null=True, blank=True)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, null=True, blank=True)
    file = models.FileField(verbose_name="файл с грязным контентом источника", upload_to=get_raw_file_path)
    hash_content = models.CharField(max_length=64, null=True, blank=True)

    created_at = models.DateTimeField(verbose_name="дата создания",
                                      auto_now_add=True)

    def clean(self):
        """Проверяет, что ровно одно из полей url или document заполнено."""
        if (self.url and self.document) or (not self.url and not self.document):
            raise ValidationError("Должно быть заполнено ровно одно из полей: url или document.")


class CleanedContent(models.Model):
    """Файл с чистым контентом источника"""
    url = models.ForeignKey(URL, on_delete=models.CASCADE, null=True, blank=True)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, null=True, blank=True)
    file = models.FileField(verbose_name="файл с чистым контентом источника", upload_to=get_cleaned_file_path)
    hash_content = models.CharField(max_length=64, null=True, blank=True)

    created_at = models.DateTimeField(verbose_name="дата создания",
                                      auto_now_add=True)
    updated_at = models.DateTimeField(verbose_name="дата обновления", auto_now=True)


@receiver(post_delete, sender=RawContent)
@receiver(post_delete, sender=CleanedContent)
def delete_content_file(sender, instance, **kwargs):
    """Удаляет файл при удалении записи RawContent или CleanedContent."""
    if instance.file and os.path.exists(instance.file.path):
        os.remove(instance.file.path)
