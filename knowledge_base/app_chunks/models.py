from django.db import models

# class Chunk(models.Model):
#     """
#     Хранит чанки контента с адаптивными параметрами.
#     """
#     source_type = models.CharField(max_length=50, help_text="Тип источника: 'url' или 'document'")
#     source_id = models.CharField(max_length=200, help_text="ID источника")
#     content = models.TextField(help_text="Текст чанка")
#     language = models.CharField(max_length=10, null=True, blank=True, help_text="Язык чанка")
#     chunking_model = models.CharField(max_length=100, help_text="Модель чанкинга, например 'recursive'")
#     chunk_size = models.IntegerField(help_text="Размер чанка")
#     chunk_overlap = models.IntegerField(default=0, help_text="Перекрытие чанков")
#     created_at = models.DateTimeField(auto_now_add=True)
#
#     def __str__(self):
#         return f"Chunk from {self.source_type} {self.source_id}"
#
#     def generate_chunks(self):
#         """Заглушка для создания чанков."""
#         from utils.chunking import generate_chunks
#         return generate_chunks(self)
#
#     class Meta:
#         verbose_name = "Chunk"
#         verbose_name_plural = "Chunks"