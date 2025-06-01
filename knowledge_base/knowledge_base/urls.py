from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter


router = DefaultRouter()
# router.register(r'parsers', ParserViewSet)
# router.register(r'urls', URLViewSet)
# router.register(r'cloud_storages', CloudStorageViewSet)
# router.register(r'documents', DocumentViewSet)
# router.register(r'chunks', ChunkViewSet)
# router.register(r'embeddings', EmbeddingViewSet)
# router.register(r'embedding_engines', EmbeddingEngineViewSet)
# router.register(r'retriever_logs', RetrieverLogViewSet)
# router.register(r'search', SearchViewSet, basename='search')

urlpatterns = [
    path('api/', include(router.urls)),
    path('admin/', admin.site.urls),
    path('sources/', include('app_sources.urls')),
]