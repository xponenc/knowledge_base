from django.urls import path

from app_chat.views import CurrentTestChunksView
from app_embeddings.views import EngineDetailView, EngineListView, EngineCreateView, EngineUpdateView, EngineDeleteView, \
    VectorizeWebsiteView, VectorizeStorageView, EngineTestTask

app_name = "embeddings"

urlpatterns = [
    path("engine/<int:pk>", EngineDetailView.as_view(), name="engine_detail"),
    path("engine", EngineListView.as_view(), name="engine_list"),
    path("engine/create", EngineCreateView.as_view(), name="engine_create"),
    path("engine/<int:pk>/update", EngineUpdateView.as_view(), name="engine_update"),
    path("engine/<int:pk>/update", EngineDeleteView.as_view(), name="engine_delete"),

    path("website/<int:pk>/vectorize", VectorizeWebsiteView.as_view(), name="website_vectorize"),
    path("storage/<str:storage_type>/<int:pk>/vectorize", VectorizeStorageView.as_view(), name="storage_vectorize"),

    path('chunks/show-current/', CurrentTestChunksView.as_view(), name='current_chuncks'),
    path('test_task', EngineTestTask.as_view(), name="test_task")
]
