from django.contrib.auth import get_user_model
from django.core.validators import MinValueValidator
from django.db import models

from app_core.models import KnowledgeBase

User = get_user_model()


class ChatSession(models.Model):
    session_key = models.CharField(verbose_name="ключ пользовательской сессии",
                                   max_length=100, db_index=True)
    kb = models.ForeignKey(KnowledgeBase, verbose_name="база знаний", on_delete=models.CASCADE,
                           related_name='chat_sessions')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ChatSession {self.id} for KB {self.kb_id} (session_key={self.session_key})"

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['session_key', 'kb'], name='unique_session_per_kb')
        ]
        indexes = [
            models.Index(fields=['session_key', 'kb'], name='session_key_kb_idx')
        ]


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

    # FAISS_DIR = os.path.join(BASE_DIR, "media", "user_questions_faiss")
    # EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="ai-forever/FRIDA")
    #
    # @receiver(post_save, sender=ChatMessage)
    # def add_new_question_to_faiss(sender, instance, created, **kwargs):
    #     if created and instance.is_user and instance.text.strip():
    #         try:
    #             db = FAISS.load_local(FAISS_DIR, EMBEDDING_MODEL, index_name="index")
    #         except Exception:
    #             db = FAISS.from_documents([Document(page_content='', metadata={})], EMBEDDING_MODEL)
    #
    #         doc = Document(page_content=instance.text, metadata={"id": instance.id})
    #         db.add_documents([doc])
    #         db.save_local(folder_path=FAISS_DIR, index_name="index")