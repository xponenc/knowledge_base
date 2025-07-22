from fastapi import APIRouter, Depends, HTTPException
from asgiref.sync import sync_to_async
from app_api.models import ApiClient
from app_users.models import CustomerProfile
from fast_api.auth import get_api_client
from fast_api.schemas.user_schemas import ProfileResponse, CustomerProfileSchema
from utils.setup_logger import setup_logger

from transliterate import translit

# Настройка логгирования
logger = setup_logger(name=__file__, log_dir="logs/fast_api", log_file="fast_api.log")

user_router = APIRouter()


@sync_to_async
def get_customer_profile(telegram_id: int):
    """
    Получает профиль пользователя по telegram_id.

    Args:
        telegram_id (int): ID пользователя в Telegram.

    Returns:
        CustomerProfile: Объект профиля пользователя или None, если профиль не найден.
    """
    logger.debug(f"Fetching profile for telegram_id={telegram_id}")
    profile = CustomerProfile.objects.select_related("user").filter(telegram_id=telegram_id).first()
    logger.debug(f"Profile found: {profile is not None}")
    return profile


@user_router.get("/{telegram_id}", response_model=ProfileResponse, summary="Получение профиля пользователя")
async def get_profile(telegram_id: int, client: ApiClient = Depends(get_api_client)):
    """
    Возвращает профиль пользователя по его Telegram ID.

    **Параметры**:
    - `telegram_id`: ID пользователя в Telegram.
    - `client`: Аутентифицированный клиент API (через заголовок Authorization).

    **Возвращает**:
    - `ProfileResponse`: Данные профиля пользователя или строка "Anonymous", если профиль не найден.

    **Ошибки**:
    - 401: Неверный или отсутствующий токен авторизации.

    **Пример запроса**:
    ```
    GET /api/user/123456789
    Authorization: Bearer your-api-key-here
    ```

    **Пример ответа**:
    ```json
    {
        "profile": {
            "user_name": "Иванов Иван Иванович",
            "catalog_user_id": 1,
            "user_name_eng": "Ivan_Ivanov",
            "email": "ivan@example.com",
            "phone": "+79991234567",
            "last_name": "Иванов",
            "first_name": "Иван",
            "middle_name": "Иванович",
            "address": "Москва, ул. Примерная, 123",
            "date_of_birth": "1990-01-01",
            "consent": true
        }
    }
    ```
    """
    logger.info(f"Fetching user profile for telegram_id={telegram_id}")
    profile = await get_customer_profile(telegram_id)

    if profile and profile.is_active:
        user_name = f"{profile.last_name or ''} {profile.first_name or ''} {profile.middle_name or ''}".strip() or "Unknown"
        user_name_eng = translit(user_name, 'ru', reversed=True)
        return ProfileResponse(
            profile=CustomerProfileSchema(
                user_name=user_name,
                profile_id=profile.id,
                user_name_eng=user_name_eng,
                # email=profile.email,
                # phone=profile.phone,
                # last_name=profile.last_name,
                # first_name=profile.first_name,
                # middle_name=profile.middle_name,
                # address=profile.address,
                # date_of_birth=profile.date_of_birth,
                # consent=profile.consent
            )
        )

    return ProfileResponse(profile="Anonymous")