from django.contrib.auth import get_user_model
from django.core.validators import RegexValidator
from django.db import models

from app_core.models import KnowledgeBase

User = get_user_model()



roles = (
    ("employee", "сотрудник"),
    ("client", "клиент"),
)


class CustomerProfile(models.Model):
    """Модель пользователя Базы знаний, не предполагает доступа к авторизации Django, но может расширить модель User"""
    user = models.OneToOneField(
        User,
        verbose_name="Пользователь системы",
        on_delete=models.CASCADE,
        related_name="customer_profile",
        null=True,
        blank=True
    )
    role = models.CharField(verbose_name="Роль", max_length=10, choices=roles, default="client")
    telegram_id = models.BigIntegerField(unique=True, verbose_name="Telegram ID", blank=True, null=True)
    last_name = models.CharField(max_length=100, verbose_name="Фамилия", blank=True)
    first_name = models.CharField(max_length=100, verbose_name="Имя", blank=True)
    middle_name = models.CharField(max_length=100, verbose_name="Отчество", blank=True, null=True)
    email = models.EmailField(verbose_name="Email", unique=True, blank=True, null=True)
    phone = models.CharField(
        max_length=20,
        verbose_name="Телефон",
        unique=True,
        blank=True,
        null=True,
        validators=[RegexValidator(regex=r'^\+?\d{10,15}$', message="Номер телефона должен содержать 10-15 цифр")]
    )
    address = models.TextField(verbose_name="Адрес доставки", blank=True, null=True)
    date_of_birth = models.DateField(verbose_name="Дата рождения", blank=True, null=True)
    consent = models.BooleanField(verbose_name="Согласие на обработку данных", default=False)
    knowledge_base = models.ForeignKey(
        KnowledgeBase,
        verbose_name="Разрешенная база знаний",
        on_delete=models.CASCADE,
        related_name="customer_profiles"
    )
    is_active = models.BooleanField(verbose_name="Активен", default=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Создан")

    def __str__(self):
        name = f"{self.last_name} {self.first_name} {self.middle_name or ''}".strip() or "Anonymous"
        return f"CustomerProfile {name} (Telegram ID: {self.telegram_id})"


    class Meta:
        verbose_name = "Пользователь базы знаний"
        verbose_name_plural = "Пользователи базы знаний"