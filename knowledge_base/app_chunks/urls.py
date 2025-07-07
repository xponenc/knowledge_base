from django.urls import path

from app_chunks.views import ChunkCreateFromURLContentView, ChunkCreateFromWebSiteView, \
    SplitterConfigView, ChunkListView, ChunkDetailView, ChunkCreateFromStorageView

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



    path("config/", SplitterConfigView.as_view(), name="splitter_config"),
]
