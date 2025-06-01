from rest_framework import viewsets
from .models import Chunk
from .serializers import ChunkSerializer

class ChunkViewSet(viewsets.ModelViewSet):
    """API для управления чанками."""
    queryset = Chunk.objects.all()
    serializer_class = ChunkSerializer