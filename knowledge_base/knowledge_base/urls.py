from django.contrib import admin
from django.conf import settings
from django.urls import path, include, re_path
from django.conf.urls.static import static
from django.views.generic import RedirectView
from rest_framework.routers import DefaultRouter

from knowledge_base.views import get_celery_task_progress

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
    path('accounts/', include('django.contrib.auth.urls')),
    path('', RedirectView.as_view(pattern_name="core:knowledgebase_list", permanent=False)),
    re_path(r'^celery-progress/', include('celery_progress.urls')),
    path('celery-progress/<str:parent_type>/<int:parent_pk>/task/<str:task_pk>/',
         get_celery_task_progress, name='celery_progress_info'),
    path('', include('app_core.urls')),
    path('sources/', include('app_sources.urls')),
    path('', include('app_parsers.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
