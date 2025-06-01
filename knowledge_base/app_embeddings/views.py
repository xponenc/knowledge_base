from rest_framework import viewsets
from .models import Embedding, EmbeddingEngine
from .serializers import EmbeddingSerializer, EmbeddingEngineSerializer

class EmbeddingViewSet(viewsets.ModelViewSet):
    """API для управления эмбеддингами."""
    queryset = Embedding.objects.all()
    serializer_class = EmbeddingSerializer

class EmbeddingEngineViewSet(viewsets.ModelViewSet):
    """API для управления движками эмбеддинга."""
    queryset = EmbeddingEngine.objects.all()
    serializer_class = EmbeddingEngineSerializer