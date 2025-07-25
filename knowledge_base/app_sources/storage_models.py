from importlib import import_module

from django.core.exceptions import ValidationError
from django.db import models
from django.contrib.auth import get_user_model
from django.db.models import UniqueConstraint, Q
from django.urls import reverse

from knowledge_base.mixin_models import TrackableModel, SoftDeleteModel
from app_core.models import KnowledgeBase

User = get_user_model()


class Storage(TrackableModel, SoftDeleteModel):
    """Абстрактная модель хранилища, предназначена для логического объединения источников и их обработки"""

    kb = models.ForeignKey(KnowledgeBase, verbose_name="база знаний", on_delete=models.CASCADE)

    name = models.CharField(max_length=200, verbose_name="Название")
    tags = models.JSONField(verbose_name="список тегов",
                            default=list,
                            blank=True,
                            help_text="Категории в формате JSON, например ['news', 'tech', ]")
    description = models.CharField(verbose_name="описание", max_length=1000, null=True, blank=True)
    configs = models.JSONField(verbose_name="конфигурации", default=dict, blank=True, null=True)

    default_retriever = models.BooleanField(verbose_name="Поиск по умолчанию", default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    soft_deleted_at = models.DateTimeField(blank=True, null=True, verbose_name="мягко удалена")

    author = models.ForeignKey(User, on_delete=models.CASCADE)

    objects = models.Manager()

    class Meta:
        abstract = True


class WebSite(Storage):
    """Модель Веб-сайта для полного парсинга сайта"""
    base_url = models.URLField(verbose_name="Основной URL сайта")
    xml_map_url = models.URLField(verbose_name="XML карта сайта", blank=True, null=True)

    class Meta:
        verbose_name = "Web Site"
        verbose_name_plural = "Web Sites"
        constraints = [
            UniqueConstraint(
                fields=['kb', 'name'],
                condition=Q(kb__isnull=False, name__isnull=False),
                name='unique_kb_and_name_for_website',
            ),
            UniqueConstraint(
                fields=['kb', 'base_url'],
                condition=Q(kb__isnull=False, base_url__isnull=False),
                name='unique_kb_and_base_url_for_website',
            )
        ]

    def get_absolute_url(self):
        return reverse("sources:website_detail", kwargs={"pk": self.pk, })

    def __str__(self):
        return f"{self.name}(web-site)"

    def clean(self):
        if self.kb and self.name:
            if WebSite.objects.filter(kb=self.kb, name=self.name).exclude(pk=self.pk).exists():
                raise ValidationError(f"Веб-сайт с именем {self.name} в базе знаний {self.kb.name} уже существует.")
        if self.kb and self.base_url:
            if WebSite.objects.filter(kb=self.kb, name=self.base_url).exclude(pk=self.pk).exists():
                raise ValidationError(f"Веб-сайт с URL {self.base_url} в базе знаний {self.kb.name} уже существует.")


class URLBatch(Storage):
    """Модель Пакете ссылок для выборочного парсинга страниц"""

    class Meta:
        verbose_name = "URL Batch"
        verbose_name_plural = "URL Batches"
        constraints = [
            UniqueConstraint(
                fields=['kb', 'name'],
                condition=Q(kb__isnull=False, name__isnull=False),
                name='unique_kb_and_name_for_urlbatch',
            ),
        ]

    def get_absolute_url(self):
        return reverse("sources:webbatch_detail", kwargs={"pk": self.id, })

    def __str__(self):
        return f"{self.name}(web-batch)"

    def clean(self):
        if self.kb and self.name:
            if URLBatch.objects.filter(kb=self.kb, name=self.name).exclude(pk=self.pk).exists():
                raise ValidationError(f"Веб-пакет с именем {self.name} в базе знаний {self.kb.name} уже существует.")


class LocalStorage(Storage):
    """Модель локального хранилища"""

    class Meta:
        verbose_name = "Local Storage"
        verbose_name_plural = "Local Storages"
        constraints = [
            UniqueConstraint(
                fields=['kb', 'name'],
                condition=Q(kb__isnull=False, name__isnull=False),
                name='unique_kb_and_name_for_local_storage',
            )
        ]

    def get_absolute_url(self):
        return reverse("sources:localstorage_detail", kwargs={"pk": self.id, })

    def __str__(self):
        return f"{self.name}(local)"

    def clean(self):
        # Только если оба поля не None — проверка на уникальность
        if self.kb and self.name:
            if LocalStorage.objects.filter(kb=self.kb, name=self.name).exclude(pk=self.pk).exists():
                raise ValidationError(f"Локальное с именем {self.name} в базе знаний {self.kb.name} уже существует.")


class CloudStorage(Storage):
    """Модель облачного хранилища файлов"""

    # Диспетчер классов хранилищ
    STORAGE_CLASSES = {
        'webdav': 'storages_external.webdav_storage.webdav_client.WebDavStorage',
        # 'google_drive': 'storages_external.google_drive.google_drive_client.GoogleDriveStorage',
        # 'dropbox': 'storages_external.dropbox.dropbox_client.DropboxStorage',
    }

    api_type = models.CharField(max_length=100, choices=((key, key) for key in STORAGE_CLASSES))
    credentials = models.JSONField(null=True, blank=True, help_text="Учетные данные")

    class Meta:
        verbose_name = "облако"
        verbose_name_plural = "облачные хранилища"
        constraints = [
            UniqueConstraint(
                fields=['kb', 'name'],
                condition=Q(kb__isnull=False, name__isnull=False),
                name='unique_kb_and_name_for_cloud_storage',
            )
        ]

    def __str__(self):
        return f"{self.name}(network)"

    def clean(self):
        if self.kb and self.name:
            if CloudStorage.objects.filter(kb=self.kb, name=self.name).exclude(pk=self.pk).exists():
                raise ValidationError(f"CloudStorage с именем {self.name} в базе знаний {self.kb.name} уже существует.")

    def get_absolute_url(self):
        return reverse("sources:cloudstorage_detail", kwargs={"pk": self.id, })

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
