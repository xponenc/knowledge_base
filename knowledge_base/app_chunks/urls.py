from django.urls import path

from app_chunks.views import ChunkCreateFromURLContentView, ChunkCreateFromWebSiteView, TestAskFridaView, ClearChatView, \
    CurrentTestChunksView, TestModelScoreView, TestModelScoreReportView, SplitterConfigView, ChunkListView, \
    ChunkDetailView, ChunkCreateFromStorageView

app_name = "chunks"

urlpatterns = [
    path("", ChunkListView.as_view(), name="chunk_list"),
    path("<int:pk>", ChunkDetailView.as_view(), name="chunk_detail"),

    path("chunk-crete/url-content/<int:url_content_pk>", ChunkCreateFromURLContentView.as_view(),
         name="create_chunks_from_url_content"),
    path("chunk-crete/website/<int:pk>", ChunkCreateFromWebSiteView.as_view(),
         name="create_chunks_from_website"),
    path("chunk-crete/<str:storage_type>/<int:storage_pk>", ChunkCreateFromStorageView.as_view(),
         name="create_chunks_from_storage"),

    path("ask/frida", TestAskFridaView.as_view(), name="ask_frida"),
    path("test-model", TestModelScoreView.as_view(), name="test_model_score"),
    path("test-model-report", TestModelScoreReportView.as_view(), name="test_model_report"),
    path('clear/frida/', ClearChatView.as_view(), name='clear_frida'),
    path('chunks/show-current/', CurrentTestChunksView.as_view(), name='current_chuncks'),

    path("config/", SplitterConfigView.as_view(), name="splitter_config"),
]
