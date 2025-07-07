from django.urls import path

from app_embeddings.views import EngineDetailView, EngineListView, EngineCreateView, EngineUpdateView, EngineDeleteView, \
    VectorizeWebsiteView
from app_embeddings.chat_views import (ChatView, TestModelScoreView, TestModelScoreReportView,
                                       ClearChatView, CurrentTestChunksView)

app_name = "embeddings"

urlpatterns = [
    path("engine/<int:pk>", EngineDetailView.as_view(), name="engine_detail"),
    path("engine", EngineListView.as_view(), name="engine_list"),
    path("engine/create", EngineCreateView.as_view(), name="engine_create"),
    path("engine/<int:pk>/update", EngineUpdateView.as_view(), name="engine_update"),
    path("engine/<int:pk>/update", EngineDeleteView.as_view(), name="engine_delete"),

    path("website/<int:pk>/vectorize", VectorizeWebsiteView.as_view(), name="website_vectorize"),

    path("chat/<int:kb_pk>", ChatView.as_view(), name="chat"),
    path('chat/clear-history', ClearChatView.as_view(), name='clear_chat'),
    path("test-model", TestModelScoreView.as_view(), name="test_model_score"),
    path("test-model-report", TestModelScoreReportView.as_view(), name="test_model_report"),

    path('chunks/show-current/', CurrentTestChunksView.as_view(), name='current_chuncks'),
]
