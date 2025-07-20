from typing import Optional

from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    """
    Схема входного сообщения от Telegram-пользователя или AI.
    """
    telegram_id: int = Field(..., description="Уникальный идентификатор пользователя Telegram")
    incoming_message_id: int = Field(..., description="ID исходного сообщения от клиента (например, Telegram)")
    text: str = Field(..., description="Текст сообщения")
    session_key: Optional[str] = Field(None, description="Опциональный ключ сессии")


class MessageOut(BaseModel):
    """
    Ответ API на сообщение: включает AI-ответ и идентификаторы сообщений.
    """
    user_message_id: int = Field(..., description="ID сохраненного пользовательского сообщения")
    ai_message_id: int = Field(..., description="ID сохраненного AI-сообщения")
    ai_text: str = Field(..., description="Ответ, сгенерированный моделью")
    incoming_message_id: int = Field(..., description="ID входящего сообщения, на которое дан ответ")
