from django.urls import path

from app_chat.views import (ChatView, TestModelScoreView, TestModelScoreReportView,
                                       ClearChatView, CurrentTestChunksView, MessageScoreView)

app_name = "chat"

urlpatterns = [
    path("<int:kb_pk>", ChatView.as_view(), name="chat"),
    path('<int:kb_pk>/clear-history', ClearChatView.as_view(), name='clear_chat'),
    path('message/<int:message_pk>/score', MessageScoreView.as_view(), name='message_score'),
    path("test-model", TestModelScoreView.as_view(), name="test_model_score"),
    path("test-model-report", TestModelScoreReportView.as_view(), name="test_model_report"),
]

