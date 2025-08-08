import secrets

from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.urls import reverse

from app_api.models import ApiClient
from knowledge_base.mixin_models import TrackableModel, SoftDeleteModel


class KnowledgeBase(TrackableModel):
    """Модель Базы Знаний"""

    llm_models = (
        ("gpt-4o-mini", "GPT-4o-mini" ),
        ("gpt-4o", "GPT-4o"),
        ("gpt-4.1", "GPT-4.1"),
        ("gpt-4.1-nano", "GPT-4.1 nano"),
        ("gpt-5", "GPT-5"),
        ("gpt-5-nano", "GPT-5 nano"),
    )

    retriever_schemas = (
        ("multichain", "MultiRetrievalQAChain"),
        ("ensemble", "EnsembleRetriever"),
    )

    def logo_upload_path(instance, filename):
        return f"kb/logos/kb_{instance.pk}_{filename}"

    engine = models.ForeignKey("app_embeddings.EmbeddingEngine", verbose_name="модель эмбеддинга",
                               on_delete=models.SET_NULL, blank=True, null=True, related_name="bases")
    llm = models.CharField(verbose_name="LLM", max_length=30, choices=llm_models, default="gpt-4o-mini")
    retriever_scheme = models.CharField(verbose_name="схема ретриверов", max_length=30, choices=retriever_schemas,
                                        default="ensemble")

    name = models.CharField(verbose_name="название", max_length=400, unique=True)
    description = models.CharField(verbose_name="описание", max_length=1000, null=True, blank=True)

    owners = models.ManyToManyField(User, verbose_name="владельцы")

    logo = models.FileField(verbose_name="логотип", blank=True, null=True, upload_to=logo_upload_path)
    system_instruction = models.CharField(verbose_name="системная инструкция", max_length=10000, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse('core:knowledgebase_detail', args=[str(self.id)])

    def is_owner_or_superuser(self, user):
        return user.is_superuser or user in self.owners.all()


@receiver(post_save, sender=KnowledgeBase)
def create_internal_api_client(sender, instance, created, **kwargs):
    """Создание обязательного внутреннего api для взаимодействия по fastapi"""
    if created:
        ApiClient.objects.create(
            name="internal api point",
            token=secrets.token_urlsafe(32),
            knowledge_base=instance
        )