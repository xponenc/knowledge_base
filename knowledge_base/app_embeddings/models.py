from enum import Enum

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import CheckConstraint, Q
from django.urls import reverse_lazy

from app_chunks.models import Chunk
from app_sources.storage_models import WebSite, URLBatch, CloudStorage, LocalStorage

User = get_user_model()


class Embedding(models.Model):
    """
    Хранит связь чанков с эмбеддингами в FAISS.
    """
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE, related_name="embedding", help_text="Связанный чанк")
    embedding_engine = models.ForeignKey('EmbeddingEngine', on_delete=models.SET_NULL, null=True,
                                         help_text="Движок эмбеддинга")
    vector_id = models.CharField(max_length=200, verbose_name="ID вектора в FAISS", unique=True)
    report = models.ForeignKey("EmbeddingsReport", verbose_name="отчет о векторизации", on_delete=models.PROTECT,
                               blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"Embedding for chunk {self.chunk.id}"

    class Meta:
        verbose_name = "Embedding"
        verbose_name_plural = "Embeddings"


class EmbeddingEngine(models.Model):
    """
    Управляет моделями эмбеддинга, мультиязычностью и fine-tuning.
    """
    name = models.CharField(verbose_name="Название движка", max_length=100, unique=True, )
    model_name = models.CharField(max_length=100, verbose_name="Имя модели",
                                  help_text="Например: sentence-transformers/mBERT")
    supports_multilingual = models.BooleanField(default=False, verbose_name="Поддержка мультиязычности")
    fine_tuning_params = models.JSONField(null=True, blank=True, verbose_name="Параметры fine-tuning", default=dict)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse_lazy("embeddings:engine_detail", kwargs={"pk": self.pk})

    class Meta:
        verbose_name = "Embedding Engine"
        verbose_name_plural = "Embedding Engines"


class ReportStatus(Enum):
    """Статус Отчета """
    CREATED = "created"
    FINISHED = "finished"
    ERROR = "error"

    @property
    def display_name(self):
        """Русское название статуса для отображения."""
        display_names = {
            "created": "В работе",
            "finished": "Завершен",
            "error": "Ошибка"
        }
        return display_names.get(self.value, self.value)


class EmbeddingsReport(models.Model):
    site = models.ForeignKey(WebSite, verbose_name="сайт", on_delete=models.CASCADE, blank=True, null=True)
    batch = models.ForeignKey(URLBatch, verbose_name="пакет", on_delete=models.CASCADE, blank=True, null=True)
    cloud_storage = models.ForeignKey(CloudStorage, verbose_name="облачное хранилище",
                                      on_delete=models.CASCADE, blank=True, null=True)
    local_storage = models.ForeignKey(LocalStorage, verbose_name="локальное хранилище",
                                      on_delete=models.CASCADE, blank=True, null=True)

    status = models.CharField(
        verbose_name="статус обработки",
        max_length=12,
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
                name="report_only_one_source_must_be_set"
            )
        ]

    def clean(self):
        super().clean()

        fields = [self.site, self.batch, self.cloud_storage, self.local_storage]
        non_null_count = sum(1 for f in fields if f is not None)

        if non_null_count != 1:
            raise ValidationError(
                "Должно быть заполнено ровно одно из полей: site, batch, cloud_storage или local_storage")