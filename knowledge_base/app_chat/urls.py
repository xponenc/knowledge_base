from django.urls import path

from app_chat.views import (ChatView, TestModelScoreReportView,
                            ClearChatView, CurrentTestChunksView, MessageScoreView, SystemChatView, ChatReportView,
                            ChatMessageDetailView, KBRandomTestView, QwenChatView, KBBulkTestView)

app_name = "chat"

urlpatterns = [
    path("message/<int:pk>", ChatMessageDetailView.as_view(), name="chat-message_detail"),

    path("<int:kb_pk>", ChatView.as_view(), name="chat"),
    path("system/<int:kb_pk>", SystemChatView.as_view(), name="system_chat"),
    path("chat_report/<int:kb_pk>", ChatReportView.as_view(), name="chat_report"),
    path('<int:kb_pk>/clear-history', ClearChatView.as_view(), name='clear_chat'),
    path('message/<int:message_pk>/score', MessageScoreView.as_view(), name='message_score'),

    path("test/<int:kb_pk>/random", KBRandomTestView.as_view(), name="kb_random_test"),
    path("test/<int:kb_pk>/bulk", KBBulkTestView.as_view(), name="kb_bulk_test"),
    path("test-model-report", TestModelScoreReportView.as_view(), name="test_model_report"),
]

