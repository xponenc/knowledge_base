from django.db import models


class RetrieverLog(models.Model):
    """
    Логирует запросы RAG для аудита и обучения.
    """
    query = models.TextField(help_text="Текст запроса")
    retrieved_chunks = models.JSONField(help_text="ID возвращенных чанков")
    embedding_model = models.CharField(max_length=100, help_text="Модель эмбеддинга")
    language = models.CharField(max_length=10, null=True, help_text="Язык запроса")
    timestamp = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, help_text="Дополнительные метаданные")

    def __str__(self):
        return f"Query: {self.query} at {self.timestamp}"

    class Meta:
        verbose_name = "Retriever Log"
        verbose_name_plural = "Retriever Logs"

