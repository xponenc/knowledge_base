from django.db import models

from app_core.models import KnowledgeBase


class ChatSession(models.Model):
    session_key = models.CharField(max_length=40, unique=True, db_index=True)
    kb = models.ForeignKey(KnowledgeBase, on_delete=models.CASCADE, related_name='chat_sessions')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ChatSession {self.id} for KB {self.kb_id} (session_key={self.session_key})"


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    is_user = models.BooleanField(default=True)
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    score = models.SmallIntegerField(null=True, blank=True)

    def __str__(self):
        author = "User" if self.is_user else "AI"
        return f"{author} message {self.id} in session {self.session_id}"