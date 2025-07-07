from django.urls import path

from app_embeddings.views import EngineDetailView, EngineListView, EngineCreateView, EngineUpdateView, EngineDeleteView, \
    VectorizeWebsiteView

app_name = "embeddings"

urlpatterns = [
    path("engine/<int:pk>", EngineDetailView.as_view(), name="engine_detail"),
    path("engine", EngineListView.as_view(), name="engine_list"),
    path("engine/create", EngineCreateView.as_view(), name="engine_create"),
    path("engine/<int:pk>/update", EngineUpdateView.as_view(), name="engine_update"),
    path("engine/<int:pk>/update", EngineDeleteView.as_view(), name="engine_delete"),

    path("website/<int:pk>/vectorize", VectorizeWebsiteView.as_view(), name="website_vectorize")
]
