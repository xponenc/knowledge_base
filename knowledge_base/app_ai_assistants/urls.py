from django.urls import path

from app_ai_assistants.views import AssistantCreateView, AssistantDetailView, AssistantUpdateView, AssistantDeleteView, \
    AssistantSaveGraphView, AssistantChatView

app_name = 'ai-assistants'

urlpatterns = [
    path("<int:kb_pk>/create", AssistantCreateView.as_view(), name="ai-assistant-crete"),
    path("<int:pk>/update", AssistantUpdateView.as_view(), name="ai-assistant-update"),
    path("<int:pk>/delete", AssistantDeleteView.as_view(), name="ai-assistant-delete"),
    path("<int:pk>/save/", AssistantSaveGraphView.as_view(), name="ai-assistant-graph-save"),

    path("<int:pk>/", AssistantDetailView.as_view(), name="ai-assistant-detail"),

    path("<int:pk>/chat", AssistantChatView.as_view(), name="ai-assistant-chat"),
]