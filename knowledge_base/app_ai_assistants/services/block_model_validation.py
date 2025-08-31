import re
from typing import Type, Literal

from django.http import QueryDict
from pydantic import BaseModel, Field, ValidationError, field_validator

from app_core.models import KnowledgeBase


class BaseBlockConfig(BaseModel):
    """Базовая модель валидации поля config модели Block"""
    verbose_name: str = Field(..., min_length=1, description="Человекочитаемое имя блока")
    model_name: str = Field(..., min_length=1, description="Название модели LLM (например, gpt-4.1-nano)")
    model_temperature: float = Field(
        0,
        description="Температура генерации (0 = детерминированный ответ, 1 = креативный)"
    )
    system_prompt: str = Field(..., min_length=1, description="Системный промпт для LLM")
    instructions: str = Field(..., min_length=1, description="Инструкции для LLM")

    @field_validator("model_name")
    def validate_model_name(cls, v):
        allowed_model_names = {m[0] for m in KnowledgeBase.llm_models}
        if v not in allowed_model_names:
            raise ValueError(f"Недопустимая модель: {v}. Разрешенные: {', '.join(allowed_model_names)}")
        return v


class ExtractorBlockConfig(BaseBlockConfig):
    """Модель валидации поля config модели Block c type extractor"""
    target: str = Field(
        ..., min_length=1,
        description="Название ключа в inputs, текст из которого будет анализировать блок"
    )
    accumulate_history: bool = Field(
        False,
        description="Флаг: если True — экстрактор будет накапливать историю результатов"
    )
    output_format: Literal["str", "list"] = Field(
        "str",
        description="Формат выходных данных: 'str' — одна строка, 'list' — список элементов"
    )


class ExpertBlockConfig(BaseBlockConfig):
    """Модель валидации поля config модели Block c type expert"""
    context_search: bool = Field(
        ...,
        description="Флаг подгрузки в LLM расширенного контекста из базы знаний"
    )


class SummaryBlockConfig(BaseModel):
    """Базовая модель валидации поля config модели summary"""
    verbose_name: str = Field(..., description="Человекочитаемое имя блока")
    model_name: str = Field(..., description="Название модели LLM (например, gpt-4.1-nano)")
    model_temperature: float = Field(
        0,
        description="Температура генерации (0 = детерминированный ответ, 1 = креативный)"
    )
    max_token_limit: int = Field(..., description="Длина саммари в токенах")


class ContainerBlockConfig(BaseModel):
    """Базовая модель валидации поля config модели Block c пустым config"""
    pass


# Маппинг типов блоков → Pydantic-модель
BLOCK_TYPE_TO_SCHEMA: dict[str, Type[BaseModel]] = {
    "extractor": ExtractorBlockConfig,
    "reformulator": BaseBlockConfig,
    "expert": ExpertBlockConfig,
    "router": BaseBlockConfig,
    "senior": BaseBlockConfig,
    "stylist": BaseBlockConfig,
    "sequential": ContainerBlockConfig,
    "parallel": ContainerBlockConfig,
    "passthrough": ContainerBlockConfig,
    "retriever": ContainerBlockConfig,
    "report": ContainerBlockConfig,
    "summary": SummaryBlockConfig,
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
    print(f"{block_type=}")
    print(f"{schema_cls=}")
    try:
        print(config)
        return schema_cls(**config)
    except ValidationError as e:
        error_text = format_pydantic_errors(e)
        raise ValueError(
            f"❌ Ошибка валидации блока [{block_name}:{block_type}]:\n{error_text}"
        ) from e
        # raise ValueError(f"❌ Ошибка валидации блока [{block_name}:{block_type}]:{e}") from e


def format_pydantic_errors(e: ValidationError) -> str:
    """
    Формирует человекочитаемый текст из ошибок Pydantic.
    """
    messages = []
    for err in e.errors():
        field_path = ".".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "Unknown error")
        value = err.get("input")
        # пример: "Поле 'model_name' — Field required (получено: {...})"
        messages.append(f"Поле '{field_path}' — {msg}. Значение: {value}")
    return "\n".join(messages)


def parse_form_keys(post_data: QueryDict) -> dict:
    """
    Превращает request.POST с ключами вида "block[1027][instructions]"
    в вложенный dict: {"block": {"1027": {"instructions": value}}}
    """
    result: dict = {}

    for key, value in post_data.items():
        # Разбиваем ключ на сегменты: block[1027][instructions] -> ["block", "1027", "instructions"]
        parts = re.split(r'\[|\]', key)
        parts = [p for p in parts if p]  # убираем пустые

        # Встраиваем значение в словарь
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return result
