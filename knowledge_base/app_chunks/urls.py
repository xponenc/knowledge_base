from django.urls import path

from app_chunks.views import ChunkCreateFromURLContentView, ChunkCreateFromWebSiteView, TestAskFridaView, ClearChatView, \
    CurrentTestChunksView

app_name = "chunks"

urlpatterns = [
    path("chunk-crete/url-content/<int:pk>", ChunkCreateFromURLContentView.as_view(), name="create_chunks_from_url_content"),
    path("chunk-crete/website/<int:pk>", ChunkCreateFromWebSiteView.as_view(), name="create_chunks_from_website"),
    path("ask/frida", TestAskFridaView.as_view(), name="ask_frida"),
    path('clear/frida/', ClearChatView.as_view(), name='clear_frida'),
    path('chunks/show-current/', CurrentTestChunksView.as_view(), name='current_chuncks'),
]
