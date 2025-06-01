from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status
from .models import RetrieverLog
from .serializers import RetrieverLogSerializer
from utils.embedding import search_in_faiss

class RetrieverLogViewSet(viewsets.ModelViewSet):
    """API для просмотра логов RAG."""
    queryset = RetrieverLog.objects.all()
    serializer_class = RetrieverLogSerializer

class SearchViewSet(viewsets.ViewSet):
    """API для поиска по базе знаний."""
    @action(detail=False, methods=['post'])
    def search(self, request):
        query = request.data.get('query')
        language = request.data.get('language')
        categories = request.data.get('categories', [])
        results = search_in_faiss(query, language, categories)
        RetrieverLog.objects.create(
            query=query,
            retrieved_chunks=[r['chunk_id'] for r in results],
            embedding_model='default',
            language=language,
            metadata={'categories': categories}
        )
        return Response(results, status=status.HTTP_200_OK)