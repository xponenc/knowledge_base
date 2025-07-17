from django.core.validators import MinValueValidator
from django.db import models

from app_core.models import KnowledgeBase


class ChatSession(models.Model):
    session_key = models.CharField(verbose_name="ключ пользовательской сессии",
                                   max_length=40, unique=True, db_index=True)
    kb = models.ForeignKey(KnowledgeBase, verbose_name="диалоговая база", on_delete=models.CASCADE, related_name='chat_sessions')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ChatSession {self.id} for KB {self.kb_id} (session_key={self.session_key})"


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, verbose_name="пользовательская сессия",
                                on_delete=models.CASCADE, related_name='messages')
    answer_for = models.OneToOneField("ChatMessage", verbose_name="ответ на", blank=True, null=True,
                                      on_delete=models.CASCADE, related_name="answer")
    is_user = models.BooleanField(verbose_name="пользователь/ai", default=True)
    text = models.TextField(verbose_name="сообщение")
    score = models.SmallIntegerField(verbose_name="оценка", null=True, blank=True,
                                     validators=[MinValueValidator(-2), MinValueValidator(2)])

    is_user_deleted = models.DateTimeField(verbose_name="удалено пользователем", blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        author = "User" if self.is_user else "AI"
        return f"{author} message {self.id} in session {self.session_id}"