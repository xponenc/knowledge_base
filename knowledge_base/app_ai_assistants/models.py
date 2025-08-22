from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import models

from rest_framework.reverse import reverse_lazy

from app_ai_assistants.services.block_model_validation import validate_block_config
from app_core.models import KnowledgeBase

User = get_user_model()


class Assistant(models.Model):
    """AI ассистент, который собирается из блоков"""
    TYPE_CHOICES = [
        ("neuro_sales", "Ассистент для продаж"),
        ("support", "Служба поддержки"),
        ("knowledge", "Knowledge Base Assistant"),
    ]
    kb = models.ForeignKey(KnowledgeBase, verbose_name="База знаний", on_delete=models.CASCADE)
    name = models.CharField(verbose_name="Название", max_length=255)
    description = models.TextField(verbose_name="Описание", blank=True)
    type = models.CharField(verbose_name="Тип помощника", max_length=50, choices=TYPE_CHOICES, default="neuro_sales")

    author = models.ForeignKey(User,  verbose_name="Автор", on_delete=models.CASCADE)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.get_type_display()})"

    def get_absolute_url(self):
        return reverse_lazy("ai-assistants:ai-assistant-detail", kwargs={"pk": self.pk})


class Block(models.Model):
    """Блок ассистента: атомарный или контейнер"""
    BLOCK_TYPES = [
        ("extractor", "Extractor"),
        ("summary", "Summary"),
        ("router", "Router"),
        ("expert", "Expert"),
        ("senior", "Senior"),
        ("stylist", "Stylist"),
        ("sequential", "Sequence (контейнер)"),
        ("parallel", "Parallel (контейнер)"),
        ("retriever", "Retriever"),
        ("passthrough", "Блок для проброса входящих inputs"),
    ]

    assistant = models.ForeignKey(Assistant, on_delete=models.CASCADE, related_name="blocks")
    name = models.CharField(max_length=255)
    block_type = models.CharField(max_length=50, choices=BLOCK_TYPES)
    config = models.JSONField(default=dict, blank=True)  # параметры блока

    def __str__(self):
        return f"{self.assistant.name} / {self.name} ({self.block_type})"

    def clean(self):
        """Валидация конфига при сохранении"""
        try:
            validate_block_config(self.block_type, self.config)
        except ValueError as e:
            raise ValidationError(str(e))

    def save(self, *args, **kwargs):
        """Жёсткая валидация при сохранении (не даст битый JSON записать в БД)"""
        self.full_clean()  # вызовет clean() + стандартные проверки
        super().save(*args, **kwargs)

class BlockConnection(models.Model):
    """Связь между блоками"""
    from_block = models.ForeignKey(Block, on_delete=models.CASCADE, related_name="outgoing_connections")
    to_block = models.ForeignKey(Block, on_delete=models.CASCADE, related_name="incoming_connections")
    order = models.PositiveIntegerField(default=0)  # порядок для последовательностей
    is_child = models.BooleanField(default=False)

    class Meta:
        ordering = ["order"]

    def __str__(self):
        return f"{self.from_block.name} → {self.to_block.name} ({self.order})"