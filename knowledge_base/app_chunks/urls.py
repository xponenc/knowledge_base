from django.urls import path

from app_chunks.views import ChunkCreateFromURLContentView, ChunkCreateFromWebSiteView

app_name = "chunks"

urlpatterns = [
    path("chunk-crete/url-content/<int:pk>", ChunkCreateFromURLContentView.as_view(), name="create_chunks_from_url_content"),
    path("chunk-crete/website/<int:pk>", ChunkCreateFromWebSiteView.as_view(), name="create_chunks_from_website")
]
