from django.urls import path

from app_core.views import KnowledgeBaseDetailView, KnowledgeBaseListView, KnowledgeBaseCreateView, \
    KnowledgeBaseUpdateView, KnowledgeBaseDeleteView

app_name = 'core'

urlpatterns = [
    path("kb/", KnowledgeBaseListView.as_view(), name="knowledgebase_list"),
    path("kb/<int:pk>", KnowledgeBaseDetailView.as_view(), name="knowledgebase_detail"),
    path("kb/create/", KnowledgeBaseCreateView.as_view(), name="knowledgebase_create"),
    path("kb/<int:pk>/update/", KnowledgeBaseUpdateView.as_view(), name="knowledgebase_update"),
    path("kb/<int:pk>/delete/", KnowledgeBaseDeleteView.as_view(), name="knowledgebase_delete"),
]