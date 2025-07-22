from pydantic import BaseModel, Field
from datetime import date
from typing import Optional, Union


class CustomerProfileSchema(BaseModel):
    """
    Схема для данных профиля пользователя.
    """
    user_name: str = Field(..., description="Полное имя пользователя (фамилия, имя, отчество)")
    profile_id: Optional[int] = Field(None, description="ID пользователя в каталоге")
    user_name_eng: str = Field(..., description="Имя пользователя на английском (без пробелов)")
    # email: Optional[str] = Field(None, description="Электронная почта пользователя")
    # phone: Optional[str] = Field(None, description="Телефон пользователя")
    # last_name: Optional[str] = Field(None, description="Фамилия пользователя")
    # first_name: Optional[str] = Field(None, description="Имя пользователя")
    # middle_name: Optional[str] = Field(None, description="Отчество пользователя")
    # address: Optional[str] = Field(None, description="Адрес пользователя")
    # date_of_birth: Optional[date] = Field(None, description="Дата рождения пользователя")
    # consent: Optional[bool] = Field(None, description="Согласие на обработку данных")


class ProfileResponse(BaseModel):
    """
    Схема для ответа API с данными профиля пользователя.
    """
    profile: Union[CustomerProfileSchema, str] = Field(
        ...,
        description="Данные профиля или 'Anonymous', если профиль не найден")
