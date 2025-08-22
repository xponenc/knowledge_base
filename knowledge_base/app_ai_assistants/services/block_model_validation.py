from typing import Type

from pydantic import BaseModel, Field, ValidationError


class BaseBlockConfig(BaseModel):
    """Базовая модель валидации поля config модели Block"""
    verbose_name: str = Field(..., description="Человекочитаемое имя блока")
    model_name: str = Field(..., description="Название модели LLM (например, gpt-4.1-nano)")
    model_temperature: float = Field(
        0,
        description="Температура генерации (0 = детерминированный ответ, 1 = креативный)"
    )
    system_prompt: str = Field(..., description="Системный промпт для LLM")
    instructions: str = Field(..., description="Инструкции для LLM")


class ExtractorBlockConfig(BaseBlockConfig):
    """Модель валидации поля config модели Block c type extractor"""
    target: str = Field(
        ...,
        description="Название ключа в inputs, текст из которого будет анализировать блок"
    )


class ExpertBlockConfig(BaseBlockConfig):
    """Модель валидации поля config модели Block c type expert"""
    context_search: bool = Field(
        ...,
        description="Флаг подгрузки в LLM расширенного контекста из базы знаний"
    )

class ContainerBlockConfig(BaseModel):
    """Базовая модель валидации поля config модели Block c пустым config"""
    pass


# Маппинг типов блоков → Pydantic-модель
BLOCK_TYPE_TO_SCHEMA: dict[str, Type[BaseModel]] = {
    "extractor": ExtractorBlockConfig,
    "expert": ExpertBlockConfig,
    "router": BaseBlockConfig,
    "senior": BaseBlockConfig,
    "stylist": BaseBlockConfig,
    "sequential": ContainerBlockConfig,
    "parallel": ContainerBlockConfig,
    "passthrough": ContainerBlockConfig,
    "retriever": ContainerBlockConfig,
    "summary": ContainerBlockConfig,
    # можно добавлять новые типы по мере надобности
}


def validate_block_config(block_type: str, config: dict, block_name: str = "") -> BaseModel:
    """
    Валидирует config блока на основе его block_type.

    Args:
        block_type (str): тип блока (extractor, expert, ...)
        config (dict): json-поле Block.config
        block_name (str): имя блока


    Returns:
        BaseModel: Pydantic-модель с валидированными данными

    Raises:
        ValidationError: если данные невалидные
        KeyError: если block_type не поддерживается
    """
    schema_cls = BLOCK_TYPE_TO_SCHEMA.get(block_type, BaseBlockConfig)
    try:
        return schema_cls(**config)
    except ValidationError as e:
        raise ValueError(f"❌ Ошибка валидации блока [{block_name}:{block_type}]: {e}") from e