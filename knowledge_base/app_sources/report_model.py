from django.contrib.auth import get_user_model
from django.db import models
from django.urls import reverse

from app_sources.storage_models import WebSite, CloudStorage
from knowledge_base.mixin_models import TrackableModel, SoftDeleteModel

User = get_user_model()


class WebSiteUpdateReport(TrackableModel):
    """Отчет по обновлению файлов Вебсайта"""

    storage = models.ForeignKey(WebSite, on_delete=models.CASCADE, related_name="reports")

    content = models.JSONField(verbose_name="отчет", default=dict)
    running_background_tasks = models.JSONField(verbose_name="выполняемые фоновые задачи по обработке отчета",
                                                default=dict)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"Отчет об обновлении вебсайта {self.storage.name}"

    def get_absolute_url(self):
        return reverse("sources:websiteupdatereport_detail", kwargs={"pk": self.pk, })


class CloudStorageUpdateReport(models.Model):
    """Отчет по обновлению файлов Облачного хранилища"""

    storage = models.ForeignKey(CloudStorage, on_delete=models.CASCADE, related_name="reports")

    content = models.JSONField(verbose_name="отчет", default=dict)
    running_background_tasks = models.JSONField(verbose_name="выполняемые фоновые задачи по обработке отчета",
                                                default=dict)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"Отчет об обновлении облачного хранилища {self.storage}"

    def get_absolute_url(self):
        return reverse("sources:cloudstorageupdatereport_detail", kwargs={"pk": self.pk, })
