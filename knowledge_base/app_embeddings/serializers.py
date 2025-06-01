from rest_framework import serializers
from .models import Embedding, EmbeddingEngine

class EmbeddingSerializer(serializers.ModelSerializer):
    """Сериализатор для модели Embedding."""
    class Meta:
        model = Embedding
        fields = ['id', 'chunk', 'embedding_engine', 'vector_id', 'created_at']

class EmbeddingEngineSerializer(serializers.ModelSerializer):
    """Сериализатор для модели EmbeddingEngine."""
    class Meta:
        model = EmbeddingEngine
        fields = ['id', 'name', 'model_name', 'supports_multilingual', 'fine_tuning_params',
                  'created_at', 'updated_at']