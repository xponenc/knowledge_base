from rest_framework import serializers
from .models import RetrieverLog

class RetrieverLogSerializer(serializers.ModelSerializer):
    """Сериализатор для модели RetrieverLog."""
    class Meta:
        model = RetrieverLog
        fields = ['id', 'query', 'retrieved_chunks', 'embedding_model', 'language',
                  'timestamp', 'metadata']