from django.contrib import admin
from django.conf import settings
from django.urls import path, include, re_path
from django.conf.urls.static import static
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
    re_path(r'^celery-progress/', include('celery_progress.urls')),
    path('sources/', include('app_sources.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
