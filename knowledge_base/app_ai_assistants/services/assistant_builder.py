import time
from functools import partial
from typing import List, Tuple

from django.db import transaction
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

from app_ai_assistants.models import Assistant, Block, BlockConnection
from app_ai_assistants.services.block_model_validation import validate_block_config
from app_core.models import KnowledgeBase
from neuro_salesman.context import search_with_retriever
from neuro_salesman.expert import build_parallel_experts
# from neuro_salesman.summary import create_summary_exact
from neuro_salesman.router import create_router_chain
from neuro_salesman.senior import create_senior_chain
from neuro_salesman.stylist import create_stylist_chain
from neuro_salesman.extractor import build_parallel_extractors
from neuro_salesman.chains.chain_logger import ChainLogger


def create_assistant_from_config(
        kb: KnowledgeBase,
        author_id: int,
        assistant_config: dict,
        roles_config: dict
) -> Tuple[Assistant | None, List[str]]:
    """
    Создаёт ассистента и его блоки на основе конфигурации.

    Функция выполняет два этапа:
    1. Валидация конфигурации блоков без сохранения в БД.
    2. Атомарное создание ассистента и всех блоков, включая связи между ними.

    Reference-блоки больше не поддерживаются, при попытке их создания
    будет выброшено исключение.

    Args:
        kb (KnowledgeBase): Экземпляр базы знаний, к которой относится ассистент.
        author_id (int): ID пользователя, который создаёт ассистента.
        assistant_config (dict): Конфигурация ассистента, включая список блоков.
        roles_config (dict): Конфигурации по ролям (extractors, experts, routers и т.д.).

    Returns:
        Tuple[Assistant | None, List[str]]:
            - assistant: созданный объект Assistant или None, если были ошибки.
            - errors: список ошибок валидации конфигурации.
    """
    errors: List[str] = []

    # ------------------------------
    # 1. Валидация всех блоков
    # ------------------------------
    def validate_block_cfg(block_cfg: dict):
        """
        Рекурсивно валидирует блок и его дочерние блоки.
        Reference-блоки пропускаются.
        """
        block_type = block_cfg["block_type"]

        if block_type != "reference":  # reference больше не нужен, но проверка оставлена
            try:
                # Валидируем блок, объединяя конфиг блока с конфигом по умолчанию для его роли
                validate_block_config(
                    block_type=block_type,
                    config=_resolve_block_config(block_cfg, roles_config),
                    block_name=block_cfg.get("name")
                )
            except ValueError as e:
                errors.append(str(e))

        # рекурсивно валидируем детей
        for child_cfg in block_cfg.get("children", []):
            validate_block_cfg(child_cfg)

    # Валидируем все верхнеуровневые блоки ассистента
    for block_cfg in assistant_config.get("blocks", []):
        validate_block_cfg(block_cfg)

    if errors:
        # Если есть ошибки, ассистент не создаём
        return None, errors

    # ------------------------------
    # 2. Создание ассистента и блоков
    # ------------------------------
    with transaction.atomic():  # атомарная транзакция, чтобы все блоки создавались как единое целое
        assistant = Assistant.objects.create(
            kb=kb,
            name=assistant_config["name"],
            description=assistant_config.get("description", ""),
            type=assistant_config["type"],
            author_id=author_id,
        )

        # Словарь для хранения всех созданных блоков по имени
        created_blocks: dict[str, Block] = {}

        def create_block(block_cfg: dict, parent: Block | None = None, order: int = 0) -> Block:
            """
            Рекурсивно создаёт блок и его детей, а также связи с родителем.
            """
            block_type = block_cfg["block_type"]
            block_name = block_cfg["name"]

            # ⚠️ reference больше не поддерживается
            if block_type == "reference":
                raise ValueError(f"Reference block '{block_name}' больше не поддерживается")

            # Создаём блок в БД
            block = Block.objects.create(
                assistant=assistant,
                name=block_name,
                block_type=block_type,
                config=_resolve_block_config(block_cfg, roles_config),
            )
            created_blocks[block_name] = block

            # Создаём связь с родительским блоком, если есть
            if parent:
                BlockConnection.objects.create(
                    from_block=parent,
                    to_block=block,
                    order=order,
                    is_child=True,
                )

            # Рекурсивно создаём детей
            for i, child_cfg in enumerate(block_cfg.get("children", [])):
                create_block(child_cfg, parent=block, order=i)

            return block

        # ------------------------------
        # 3. Верхнеуровневая линейка блоков
        # ------------------------------
        prev_block = None
        for i, block_cfg in enumerate(assistant_config.get("blocks", [])):
            block = create_block(block_cfg, parent=None, order=i)
            # Создаём связи между верхнеуровневыми блоками для линейного потока
            if prev_block:
                BlockConnection.objects.create(
                    from_block=prev_block,
                    to_block=block,
                    order=i,
                    is_child=False,
                )
            prev_block = block

    return assistant, []


def _resolve_block_config(block_cfg, roles_config):
    """
    Находит конфиг по умолчанию для блока из roles_config и объединяет его с block_cfg.
    """
    block_type = block_cfg["block_type"]
    print(f"{block_type=}")

    mapping = {
        "extractor": roles_config.get("EXTRACTORS", {}),
        "router": roles_config.get("ROUTERS", {}),
        "expert": roles_config.get("EXPERTS", {}),
        "senior": roles_config.get("SENIOR", {}),
        "stylist": roles_config.get("STYLIST", {}),
        "summary": roles_config.get("SUMMARY", {}),
        "reformulator": roles_config.get("REFORMULATE", {}),
    }

    default_config = mapping.get(block_type, {})

    print(f"{block_cfg=}")

    # Экстракторы/эксперты/роутеры — отдельные словари
    if "name" in block_cfg and block_type in {"extractor", "expert", "router"}:
        default_config = default_config.get(block_cfg["name"])
    print(default_config)
    # if "name" in block_cfg and block_type in {"sequence", "parallel", }:
    #     default_config = default_config.get(block_cfg["name"])

    # if block_type == "extractor" and "name" in block_cfg:
    #     default_config = default_config.get(block_cfg["name"])
    #     print(f"{default_config=}")
    #
    # elif block_type == "expert" and "name" in block_cfg:
    #     default_config = default_config.get(block_cfg["name"])
    # elif block_type == "router" and "name" in block_cfg:
    #     default_config = default_config.get(block_cfg["name"])

    return {**(default_config or {}), **block_cfg.get("config", {})}
