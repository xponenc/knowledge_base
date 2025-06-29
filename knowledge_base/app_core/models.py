from django.contrib.auth.models import User
from django.db import models
from django.urls import reverse

from knowledge_base.mixin_models import TrackableModel, SoftDeleteModel


class KnowledgeBase(TrackableModel):
    """Модель Базы Знаний"""

    def logo_upload_path(instance, filename):
        return f"kb/logos/kb_{instance.pk}_{filename}"

    name = models.CharField(verbose_name="название", max_length=400, unique=True)
    description = models.CharField(verbose_name="описание", max_length=1000, null=True, blank=True)

    owners = models.ManyToManyField(User, verbose_name="владельцы")

    logo = models.FileField(verbose_name="логотип", blank=True, null=True, upload_to=logo_upload_path)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_absolute_url(self):
        return reverse('core:knowledgebase_detail', args=[str(self.id)])

    def is_owner_or_superuser(self, user):
        return user.is_superuser or user in self.owners.all()
