from django.db import models
from chunks.models import Chunk


class Embedding(models.Model):
    """
    Хранит связь чанков с эмбеддингами в FAISS.
    """
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE, related_name="embeddings", help_text="Связанный чанк")
    embedding_engine = models.ForeignKey('EmbeddingEngine', on_delete=models.SET_NULL, null=True,
                                         help_text="Движок эмбеддинга")
    vector_id = models.CharField(max_length=200, help_text="ID вектора в FAISS")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Embedding for chunk {self.chunk.id}"

    def generate_embedding(self):
        """Заглушка для создания эмбеддинга."""
        from utils.embedding import generate_embedding
        return generate_embedding(self)

    class Meta:
        verbose_name = "Embedding"
        verbose_name_plural = "Embeddings"


class EmbeddingEngine(models.Model):
    """
    Управляет моделями эмбеддинга, мультиязычностью и fine-tuning.
    """
    name = models.CharField(max_length=100, unique=True, help_text="Название движка")
    model_name = models.CharField(max_length=100, help_text="Имя модели, например 'sentence-transformers/mBERT'")
    supports_multilingual = models.BooleanField(default=False, help_text="Поддержка мультиязычности")
    fine_tuning_params = models.JSONField(null=True, blank=True, help_text="Параметры fine-tuning")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Embedding Engine"
        verbose_name_plural = "Embedding Engines"


from django.db import models

# Create your models here.
