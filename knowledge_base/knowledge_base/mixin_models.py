from django.db import models
from django.utils import timezone
from django.core.serializers.json import DjangoJSONEncoder

from django.contrib.auth import get_user_model
User = get_user_model()


class TrackableModel(models.Model):
    """
    Абстрактная модель для:
    - логирования изменений (`history`)
    - поддержки M2M сравнения (например, `owners`)
    """

    history = models.JSONField(default=list, verbose_name="лог изменений", encoder=DjangoJSONEncoder)

    class Meta:
        abstract = True

    # def soft_delete(self, user):
    #     """
    #     Устанавливает флаг мягкого удаления и логирует событие.
    #     """
    #     self.soft_deleted_at = timezone.now()
    #     self.log_history(user, 'soft_delete', {})
    #     self.save()

    def log_history(self, user, action, changes):
        """
        Добавляет новую запись в историю изменений.
        :param user: пользователь, вызвавший изменение
        :param action: строка ("create", "update", "soft_delete")
        :param changes: словарь с изменёнными полями
        """
        self.history.append({
            "timestamp": timezone.now().isoformat(),
            "user_id": user.id,
            "username": user.username,
            "action": action,
            "changes": changes,
        })

    def compare_m2m(self, field_name, old_qs, new_qs):
        """
        Сравнивает M2M-поля, возвращает словарь с добавленными/удалёнными элементами.
        """
        old_ids = set(old_qs.values_list('id', flat=True))
        new_ids = set(new_qs.values_list('id', flat=True))
        added_ids = new_ids - old_ids
        removed_ids = old_ids - new_ids
        print(f"added: {list(User.objects.filter(id__in=added_ids).values('id', 'username'))}\n"
              f"removed {list(User.objects.filter(id__in=removed_ids).values('id', 'username'))}")

        if added_ids or removed_ids:
            return {
                field_name: {
                    "added": list(User.objects.filter(id__in=added_ids).values('id', 'username')),
                    "removed": list(User.objects.filter(id__in=removed_ids).values('id', 'username')),
                }
            }
        return {}

    def save_with_log(self, user):
        """
        Сохраняет объект с логированием всех изменений и сравнениями M2M.
        """
        changes = {}
        is_update = self.pk is not None

        if is_update:
            old = self.__class__.objects.filter(pk=self.pk).first()
            if old:
                for field in self._meta.fields:
                    name = field.name
                    if name in ['updated_at']:  # игнорируем технические поля
                        continue
                    old_val = getattr(old, name)
                    new_val = getattr(self, name)
                    if old_val != new_val:
                        changes[name] = {"from": old_val, "to": new_val}

        super().save()  # сохраняем до m2m, чтобы id был

        if is_update:
            for field in self._meta.many_to_many:
                print(field.name)
                if field.name == 'owners':
                    print(old)
                    old_m2m = getattr(old, field.name)
                    print(old_m2m)
                    new_m2m = getattr(self, field.name)
                    print(new_m2m)
                    m2m_changes = self.compare_m2m(field.name, old_m2m.all(), new_m2m.all())
                    changes.update(m2m_changes)

        action = 'update' if is_update else 'create'
        if changes or not is_update:
            self.log_history(user, action, changes)
            super().save()  # ещё раз сохраняем с обновлённой историей


class SoftDeleteQuerySet(models.QuerySet):
    def delete(self, *args, **kwargs):
        """Мягкое удаление для массовых операций."""
        if kwargs.pop('hard_delete', False):
            super().delete(*args, **kwargs)
        else:
            self.update(is_deleted=True, deleted_at=timezone.now())

    def all(self, include_deleted=False):
        """Переопределяем all() для исключения удаленных объектов."""
        if include_deleted:
            return super().all()
        return self.filter(is_deleted=False)

    def deleted(self):
        """Метод для получения только удаленных объектов."""
        return self.filter(is_deleted=True)


class SoftDeleteManager(models.Manager):
    def get_queryset(self):
        return SoftDeleteQuerySet(self.model, using=self._db)

    def all(self, include_deleted=False):
        return self.get_queryset().all(include_deleted=include_deleted)

    def deleted(self):
        return self.get_queryset().deleted()

    def hard_delete(self):
        """Полное удаление через менеджер."""
        self.get_queryset().hard_delete()


class SoftDeleteModel(models.Model):
    """Абстрактная модель мягкого удаления объектов
    Использование:
    Обычное удаление
        obj = MyModel.objects.get(id=1)
        obj.delete() # Мягкое удаление (is_deleted=True, deleted_at=установлено)

    Массовое удаление
        MyModel.objects.all().delete() # Мягкое удаление для всех объектов

    Полное удаление (для суперпользователя)
        obj.delete(hard_delete=True) # Полное удаление
        MyModel.objects.all().hard_delete() # Полное удаление всех объектов

    Фильтрация в коде
        MyModel.objects.all() # Только неудаленные
        MyModel.objects.all(include_deleted=True) # Все, включая удаленные
        MyModel.objects.deleted() # Только удаленные

    """
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        abstract = True
        indexes = [models.Index(fields=['is_deleted'])]

    objects = SoftDeleteManager()

    def delete(self, *args, **kwargs):
        """Переопределяем метод delete для мягкого удаления."""
        if kwargs.pop('hard_delete', False):
            # Полное удаление для суперпользователя
            super().delete(*args, **kwargs)
        else:
            # Мягкое удаление
            self.is_deleted = True
            self.deleted_at = timezone.now()
            self.save()

            # Мягкое удаление связанных объектов
            self._soft_delete_related()

    def _soft_delete_related(self):
        """Мягкое удаление связанных объектов."""
        for related in self._meta.related_objects:
            if related.on_delete == models.CASCADE:
                accessor_name = related.get_accessor_name()
                related_objects = getattr(self, accessor_name).all()
                for obj in related_objects:
                    if isinstance(obj, SoftDeleteModel):
                        obj.delete()  # Вызываем мягкое удаление для связанных объектов

    @classmethod
    def hard_delete(cls, queryset):
        """Метод для полного удаления набора объектов."""
        queryset.delete()

    def restore(self):
        """Восстановление мягко удаленного объекта."""
        self.is_deleted = False
        self.deleted_at = None
        self.save()

        # Восстановление связанных объектов
        for related in self._meta.related_objects:
            if related.on_delete == models.CASCADE:
                accessor_name = related.get_accessor_name()
                related_objects = getattr(self, accessor_name).all()
                for obj in related_objects:
                    if isinstance(obj, SoftDeleteModel):
                        obj.restore()