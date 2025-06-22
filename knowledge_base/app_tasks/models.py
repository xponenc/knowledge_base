from enum import Enum

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import models
from django.urls import reverse_lazy

from app_sources.content_models import URLContent, RawContent, CleanedContent
from app_sources.source_models import URL, NetworkDocument, SourceStatus, LocalDocument

User = get_user_model()


class TaskStatus(Enum):
    CREATED = "created"
    SOLVED = "solved"
    REJECTED = "rejected"

    @property
    def display_name(self):
        """Русское название статуса для отображения."""
        display_names = {
            "created": "ожидает",
            "solved": "решена",
            "rejected": "отклонена",
        }
        return display_names.get(self.value, self.value)


class ContentComparison(models.Model):
    """Сравнение старого и нового контента"""
    CONTENT_TYPES = (
        ('url_content', 'URL Content'),
        ('raw_content', 'Raw Content'),
        ('cleaned_content', 'Cleaned Content'),
    )
    content_type = models.CharField(max_length=20, choices=CONTENT_TYPES)

    old_url_content = models.ForeignKey(URLContent, related_name='old_url_comparisons',
                                        on_delete=models.CASCADE, null=True, blank=True)
    new_url_content = models.ForeignKey(URLContent, related_name='new_url_comparisons',
                                        on_delete=models.CASCADE, null=True, blank=True)
    old_raw_content = models.ForeignKey(RawContent, related_name='old_raw_comparisons',
                                        on_delete=models.CASCADE, null=True, blank=True)
    new_raw_content = models.ForeignKey(RawContent, related_name='new_raw_comparisons',
                                        on_delete=models.CASCADE, null=True, blank=True)
    old_cleaned_content = models.ForeignKey('app_sources.CleanedContent',
                                            related_name='old_cleaned_comparisons',
                                            on_delete=models.CASCADE, null=True, blank=True)
    new_cleaned_content = models.ForeignKey('app_sources.CleanedContent',
                                            related_name='new_cleaned_comparisons',
                                            on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        verbose_name = "Content Comparison"
        verbose_name_plural = "Content Comparisons"
        indexes = [
            models.Index(fields=['content_type']),
            models.Index(fields=['old_url_content']),
            models.Index(fields=['new_url_content']),
            models.Index(fields=['old_raw_content']),
            models.Index(fields=['new_raw_content']),
            models.Index(fields=['old_cleaned_content']),
            models.Index(fields=['new_cleaned_content']),
        ]

    def clean(self):
        valid_fields = {
            'url_content': ('old_url_content', 'new_url_content'),
            'raw_content': ('old_raw_content', 'new_raw_content'),
            'cleaned_content': ('old_cleaned_content', 'new_cleaned_content'),
        }
        expected_fields = valid_fields.get(self.content_type)
        for field in ('old_url_content', 'new_url_content', 'old_raw_content', 'new_raw_content',
                      'old_cleaned_content', 'new_cleaned_content'):
            value = getattr(self, field)
            if field in expected_fields:
                if value is None:
                    raise ValidationError(f"Поле {field} должно быть заполнено для {self.content_type}.")
            else:
                if value is not None:
                    raise ValidationError(f"Поле {field} должно быть пустым для {self.content_type}.")


class TaskForSource(models.Model):
    """Задача на изменение контента источника"""
    url = models.ForeignKey(
        'app_sources.URL', on_delete=models.CASCADE, related_name='tasks', null=True, blank=True
    )
    network_document = models.ForeignKey(
        'app_sources.NetworkDocument', on_delete=models.CASCADE,
        related_name='tasks', null=True, blank=True
    )
    local_document = models.ForeignKey(
        'app_sources.LocalDocument', on_delete=models.CASCADE,
        related_name='tasks', null=True, blank=True
    )
    comparison = models.OneToOneField(
        ContentComparison, related_name='task', on_delete=models.CASCADE, null=True, blank=True
    )

    status = models.CharField(
        verbose_name="статус задачи",
        max_length=10,
        choices=[(status.value, status.display_name) for status in TaskStatus],
        default=TaskStatus.CREATED.value
    )
    source_previous_status = models.CharField(
        verbose_name="Старый статус источника",
        max_length=10,
        choices=[(status.value, status.display_name) for status in SourceStatus],
    )
    source_next_status = models.CharField(
        verbose_name="Следующий статус источника",
        max_length=10,
        choices=[(status.value, status.display_name) for status in SourceStatus],
    )
    description = models.CharField(verbose_name="описание задачи", max_length=300)
    result = models.CharField(verbose_name="результат задачи", max_length=300, blank=True)

    executor = models.ForeignKey(User, verbose_name="исполнитель", on_delete=models.SET_NULL, blank=True, null=True)
    created_at = models.DateTimeField(verbose_name="дата создания", auto_now_add=True)
    finished_at = models.DateTimeField(verbose_name="дата завершения", blank=True, null=True)

    class Meta:
        verbose_name = "Task for Source"
        verbose_name_plural = "Tasks for Sources"
        indexes = [
            models.Index(fields=['url', 'status']),
            models.Index(fields=['network_document', 'status']),
            models.Index(fields=['local_document', 'status']),
            models.Index(fields=['created_at']),
        ]

    def clean(self):
        # Проверка, что ровно один источник указан
        sources = [self.url, self.network_document, self.local_document]
        if sum(bool(s) for s in sources) != 1:
            raise ValidationError("Должен быть указан ровно один источник: URL, NetworkDocument или LocalDocument.")
        # Проверка, что source_next_status не WAIT
        if self.source_next_status == SourceStatus.WAIT.value:
            raise ValidationError("Следующий статус не может быть WAIT.")
        # Проверка соответствия comparison и источника
        if self.comparison:
            source = self.get_source()
            if isinstance(source, URL) and self.comparison.content_type != 'url_content':
                raise ValidationError("Comparison для URL должно быть типа 'url_content'.")
            if (isinstance(source, (NetworkDocument, LocalDocument)) and self.comparison.content_type
                    not in ('raw_content', 'cleaned_content')):
                raise ValidationError("Comparison для NetworkDocument или LocalDocument должно"
                                      " быть типа 'raw_content' или 'cleaned_content'.")

    def get_source(self):
        """Возвращает источник (URL, NetworkDocument или LocalDocument)"""
        return self.url or self.network_document or self.local_document

    def save(self, *args, **kwargs):
        self.clean()
        is_new = self.pk is None
        source = self.get_source()

        # Устанавливаем статус WAIT при создании задачи, если источник не в WAIT
        if is_new and source.status != SourceStatus.WAIT.value:
            source.status = SourceStatus.WAIT.value
            source.save()

        super().save(*args, **kwargs)

        # Обновляем статус источника только если нет других открытых задач
        if self.status in [TaskStatus.SOLVED.value, TaskStatus.REJECTED.value]:
            open_tasks = TaskForSource.objects.filter(
                models.Q(url=source) | models.Q(network_document=source) | models.Q(local_document=source),
                status=TaskStatus.CREATED.value
            ).exclude(pk=self.pk).exists()
            if not open_tasks:
                source.status = self.source_next_status
                source.save()

    def get_absolute_url(self):
        return reverse_lazy("tasks:task_for_source_detail", kwargs={"pk": self.pk})