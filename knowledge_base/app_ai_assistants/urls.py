from django.urls import path

from app_ai_assistants.views import AssistantCreateView, AssistantDetailView, AssistantUpdateView, AssistantDeleteView, \
    AssistantSaveGraphView, AssistantChatView, BlockConfigUpdateView, AssistantSystemChatView

app_name = 'ai-assistants'

urlpatterns = [
    path("<int:kb_pk>/create", AssistantCreateView.as_view(), name="ai-assistant-crete"),
    path("<int:pk>/update", AssistantUpdateView.as_view(), name="ai-assistant-update"),
    path("<int:pk>/delete", AssistantDeleteView.as_view(), name="ai-assistant-delete"),
    path("<int:pk>/save/", AssistantSaveGraphView.as_view(), name="ai-assistant-graph-save"),

    path("<int:pk>/", AssistantDetailView.as_view(), name="ai-assistant-detail"),

    path("block/<int:pk>/config/update", BlockConfigUpdateView.as_view(), name="block-config-update"),

    path("<int:pk>/chat", AssistantChatView.as_view(), name="ai-assistant-chat"),
    path("<int:pk>/system-chat", AssistantSystemChatView.as_view(), name="ai-assistant-system-chat"),
]