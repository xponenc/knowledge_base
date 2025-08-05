from django.db import models


class ApiClient(models.Model):
    name = models.CharField(max_length=100)
    token = models.CharField(max_length=255, unique=True)
    knowledge_base = models.ForeignKey("app_core.KnowledgeBase", on_delete=models.CASCADE)

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"API Client {self.name} (KB: {self.knowledge_base.name})"

