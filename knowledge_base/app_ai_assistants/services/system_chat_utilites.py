from collections import defaultdict
from typing import Any, Type

from django.db.models import Prefetch

from app_ai_assistants.models import BlockConnection
from app_ai_assistants.services.block_model_validation import BLOCK_TYPE_TO_SCHEMA, BaseBlockConfig

# Константа: какие ключи конфига будем показывать в runtime-форме
RUNTIME_CONFIG_KEYS = ["model_name", "model_temperature", "system_prompt", "instructions", "max_token_limit"]

#
# def prepare_runtime_form_for_block(block) -> dict[str, dict[str, any]]:
#     runtime_form = {}
#     schema_cls = BLOCK_TYPE_TO_SCHEMA.get(block.block_type, BaseBlockConfig)
#
#     for key in RUNTIME_CONFIG_KEYS:
#         if key not in block.config:
#             continue
#
#         value = block.config[key]
#
#         # Pydantic 2: __fields__ → ModelField; в нём type_ и required
#         model_field = schema_cls.model_fields[key]  # для Pydantic 2
#         field_type = model_field.annotation
#         input_type = "number" if field_type in (int, float) else "text"
#         required = model_field.is_required  # True если Field(...)
#
#         runtime_form[key] = {
#             "value": value,
#             "id": f"block-{block.id}-{key}",
#             "name": f"block[{block.id}][{key}]",
#             "type": input_type,
#             "required": required,
#         }
#
#     return runtime_form
#
#
# def build_assistant_runtime_forms(assistant) -> list[dict[str, Any]]:
#     """
#     Строим иерархическую структуру ассистента с runtime-формами.
#     """
#     # 1. Загружаем блоки и связи одним запросом
#     blocks = {
#         b.id: b
#         for b in assistant.blocks.all().prefetch_related(
#             Prefetch("outgoing_connections", queryset=BlockConnection.objects.all())
#         )
#     }
#
#     children_map = defaultdict(list)
#     sequence_map = defaultdict(list)
#
#     for b in blocks.values():
#         for conn in b.outgoing_connections.all():
#             if conn.is_child:
#                 children_map[b.id].append(conn.to_block_id)
#             else:
#                 sequence_map[b.id].append(conn.to_block_id)
#
#     def build_tree(block_id):
#         block = blocks[block_id]
#         node = {
#             "id": block.id,
#             "name": block.name,
#             "block_type": block.block_type,
#             # runtime-форма только для выбранных ключей
#             "runtime_form":  prepare_runtime_form_for_block(block) if block.config else None,
#             "children": [build_tree(cid) for cid in children_map.get(block_id, [])],
#         }
#         return node
#
#     # Стартовые блоки (без входящих)
#     start_blocks = [b.id for b in blocks.values() if not b.incoming_connections.exists()]
#
#     result = []
#     for start_id in start_blocks:
#         cur_id = start_id
#         while cur_id:
#             node = build_tree(cur_id)
#             result.append(node)
#             # переход по цепочке
#             next_ids = sequence_map.get(cur_id, [])
#             cur_id = next_ids[0] if next_ids else None
#
#     return result


# ---------------------------
# Функции подготовки runtime-форм
# ---------------------------

def prepare_runtime_form_for_block(block) -> dict[str, dict[str, Any]]:
    """
    Генерируем "runtime-form" для фронта для одного блока.

    Args:
        block: объект модели Block с полем `config` (dict)

    Returns:
        dict[str, dict[str, Any]]: структура с полями для фронта.
            Каждое поле содержит:
                - value: текущее значение из block.config
                - id: уникальный HTML id
                - name: HTML name для формы
                - type: HTML input type ("text" или "number")
                - required: True/False (обязательное поле)
    """
    runtime_form = {}

    # Получаем Pydantic-схему для данного типа блока
    schema_cls: Type[BaseBlockConfig] = BLOCK_TYPE_TO_SCHEMA.get(
        block.block_type, BaseBlockConfig
    )

    for key in RUNTIME_CONFIG_KEYS:
        if key not in block.config:
            continue

        value = block.config[key]
        # Берем метаданные поля из Pydantic-модели
        model_field = schema_cls.model_fields.get(key)
        if not model_field:
            continue  # поле не описано в схеме, пропускаем

        # Определяем HTML type для input
        field_type = model_field.annotation
        input_type = "number" if field_type in (int, float) else "text"

        # Определяем required по Pydantic
        required = model_field.is_required

        runtime_form[key] = {
            "value": value.replace("\r\n", "\n").strip() if isinstance(value, str) else value,
            "id": f"block-{block.id}-{key}",
            "name": f"{key}",
            "type": input_type,
            "required": required,
        }

    return runtime_form


def build_assistant_runtime_forms(assistant) -> list[dict[str, Any]]:
    """
    Строим иерархическую структуру ассистента с runtime-формами для фронта.

    Args:
        assistant: объект ассистента, у которого есть blocks

    Returns:
        list[dict]: список блоков с вложенными children и runtime_form
    """
    # Загружаем блоки одним запросом и их связи
    blocks = {
        b.id: b
        for b in assistant.blocks.all().prefetch_related(
            Prefetch("outgoing_connections", queryset=BlockConnection.objects.all())
        )
    }

    # Создаем карты связей
    children_map = defaultdict(list)   # иерархические "дети"
    sequence_map = defaultdict(list)   # последовательные блоки

    # Заполняем карты связей
    for b in blocks.values():
        # outgoing_connections — RelatedManager, нужно вызывать .all()
        for conn in b.outgoing_connections.all():
            if conn.is_child:
                children_map[b.id].append(conn.to_block_id)
            else:
                sequence_map[b.id].append(conn.to_block_id)

    def build_tree(block_id: int) -> dict[str, Any]:
        """
        Рекурсивно строим структуру блока с children.
        """
        block = blocks[block_id]
        node = {
            "id": block.id,
            "name": block.name,
            "block_type": block.block_type,
            "runtime_form": prepare_runtime_form_for_block(block) if block.config else None,
            "children": [build_tree(cid) for cid in children_map.get(block_id, [])],
        }
        return node

    # Определяем стартовые блоки (без входящих)
    start_blocks = [b.id for b in blocks.values() if not b.incoming_connections.exists()]

    # Строим результат, обходя цепочки
    result = []
    for start_id in start_blocks:
        cur_id = start_id
        while cur_id:
            node = build_tree(cur_id)
            result.append(node)
            # переход по цепочке (sequence_map)
            next_ids = sequence_map.get(cur_id, [])
            cur_id = next_ids[0] if next_ids else None

    return result
