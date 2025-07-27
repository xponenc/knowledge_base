from django.contrib.auth import get_user_model
from django.core.validators import MinValueValidator
from django.db import models

from app_core.models import KnowledgeBase

User = get_user_model()


class ChatSession(models.Model):
    session_key = models.CharField(verbose_name="ключ пользовательской сессии",
                                   max_length=100, unique=True, db_index=True)
    kb = models.ForeignKey(KnowledgeBase, verbose_name="база знаний", on_delete=models.CASCADE,
                           related_name='chat_sessions')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ChatSession {self.id} for KB {self.kb_id} (session_key={self.session_key})"


class TelegramSession(models.Model):
    telegram_id = models.BigIntegerField(verbose_name="id telegram пользователя", null=True, blank=True)
    user = models.ForeignKey(User, verbose_name="Пользователь системы", on_delete=models.CASCADE,
                             blank=True, null=True)
    kb = models.ForeignKey(KnowledgeBase, verbose_name="база знаний", on_delete=models.CASCADE,
                           related_name='telegram_sessions')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"TelegramSession {self.telegram_id} for KB {self.kb_id}"


class ChatMessage(models.Model):
    web_session = models.ForeignKey(ChatSession, verbose_name="пользовательская web сессия",
                                    on_delete=models.CASCADE, related_name='messages', blank=True, null=True)
    t_session = models.ForeignKey(TelegramSession, verbose_name="пользовательская telegram сессия",
                                  on_delete=models.CASCADE, related_name='messages', blank=True, null=True)

    answer_for = models.OneToOneField("ChatMessage", verbose_name="ответ на", blank=True, null=True,
                                      on_delete=models.CASCADE, related_name="answer")
    is_user = models.BooleanField(verbose_name="пользователь/ai", default=True)
    text = models.TextField(verbose_name="сообщение")
    score = models.SmallIntegerField(verbose_name="оценка", null=True, blank=True,
                                     validators=[MinValueValidator(-2), MinValueValidator(2)])

    is_user_deleted = models.DateTimeField(verbose_name="удалено пользователем", blank=True, null=True)
    extended_log = models.JSONField(verbose_name="расширенный лог", default=dict)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        author = "User" if self.is_user else "AI"
        session = f"web {self.web_session_id}" if self.web_session else f"telegram {self.t_session_id}"
        return f"{author} message {self.id} in session {session}"
