from functools import partial
from typing import Dict, Any, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough

from app_ai_assistants.models import Assistant
from app_ai_assistants.services.block_model_validation import validate_block_config
from app_ai_assistants.services.visualization import build_assistant_structure
from neuro_salesman.expert import make_expert_chain, create_expert_chain
from neuro_salesman.extractor import make_extractor_chain
from neuro_salesman.reformulate import make_reformulator_chain
from neuro_salesman.retrivers import ensemble_retriever_search
from neuro_salesman.router import create_router_chain
from neuro_salesman.senior import create_senior_chain
from neuro_salesman.stylist import create_stylist_chain
from neuro_salesman.summary import create_extractors_report, update_session_summary
from neuro_salesman.utils import print_dict_structure
from utils.setup_logger import setup_logger

logger = setup_logger(name=__name__, log_dir="logs/app_ai_assistants", log_file="assistant.log")


class RuntimeConfigError(Exception):
    """Ошибка валидации runtime-конфига"""
    pass


def build_runnable_from_block(
        block: dict,
        session_info: str,
        extractors: Dict,
        api_key: str,
):
    btype = block["block_type"]
    config = block.get("config", {})  # теперь всегда тащим config

    # простые блоки
    if btype == "extractor":
        return make_extractor_chain(
            chain_name=block["name"],
            chain_config=config,
            session_info=session_info,
        )
    if btype == "reformulator":
        return make_reformulator_chain(
            chain_name=block["name"],
            chain_config=config,
            session_info=session_info,
        )
    elif btype == "report":

        return create_extractors_report(
            chain_name="Extractor reports",
            session_info=session_info,
            extractors=extractors
        )
    elif btype == "summary":

        return update_session_summary(
            chain_name=block["name"],
            session_info=session_info,
            chain_config=config
        )
    elif btype == "router":
        return create_router_chain(
            chain_name=block["name"],
            chain_config=config,
            session_info=session_info,
        )
    elif btype == "expert":
        return create_expert_chain(
            chain_name=block["name"],
            chain_config=config,
            session_info=session_info,
        )
    elif btype == "senior":
        return create_senior_chain(
            chain_name="senior",
            chain_config=config,
            session_info=session_info,
        )
    elif btype == "stylist":
        return create_stylist_chain(
            chain_name="stylist",
            chain_config=config,
            session_info=session_info,
        )
    elif btype == "retriever":
        return ensemble_retriever_search(api_key=api_key)
    elif btype == "passthrough":
        return RunnablePassthrough()

    # контейнеры
    elif btype == "sequential":
        children = [build_runnable_from_block(
            block=c,
            session_info=session_info,
            extractors=extractors,
            api_key=api_key,
        ) for c in block["children"]]
        return RunnableSequence(*children)

    elif btype == "parallel":
        children = {
            c["name"]: build_runnable_from_block(
                block=c,
                session_info=session_info,
                extractors=extractors,
                api_key=api_key,
            )
            for c in block["children"]
        }

        # Добавляем возможность передать исходные входные данные
        children['original_inputs'] = RunnablePassthrough()

        return RunnableParallel(**children)

    else:
        raise ValueError(f"Неизвестный тип блока: {btype}")


# def build_assistant_chain(assistant: Assistant, debug_mode: bool = False):
#     structure = build_assistant_structure(assistant)  # твоя функция
#     chain_parts = [build_runnable_from_block(b, debug_mode) for b in structure]
#
#     # если верхний уровень несколько блоков — объединяем в последовательность
#     if len(chain_parts) == 1:
#         return chain_parts[0]
#     return RunnableSequence(*chain_parts)


def build_assistant_chain(
        assistant: Assistant,
        session_type: str,
        session_id: str,
        roles_config: dict,
        api_key: str,
        runtime_configs: dict | None = None,

):
    """
    Собираем цепочку для ассистента.
    runtime_configs — словарь вида {block_id: {...}} для переопределения конфигов на лету.
    """
    session_info = f"{session_type}:{session_id}"
    runtime_configs = runtime_configs or {}
    structure = build_assistant_structure(assistant)
    extractors = roles_config.get("EXTRACTORS", {})
    extractors = {name: config.get("verbose_name") for name, config in extractors.items()}

    def apply_runtime_config(block: dict):
        """Рекурсивно подменяет config и валидирует runtime-конфиги."""
        block_id = block["id"]

        # если есть runtime-config для этого блока — валидируем и подменяем
        if block_id in runtime_configs:
            candidate_config = runtime_configs[block_id]
            try:
                validate_block_config(block["block_type"], candidate_config)
            except ValueError as e:
                raise RuntimeConfigError(
                    f"Ошибка валидации runtime-конфига для блока {block['name']} ({block['block_type']}): {e}"
                )
            block["config"] = candidate_config

        # рекурсивно для детей
        for child in block.get("children", []):
            apply_runtime_config(child)

    chain_parts = []

    for block in structure:
        apply_runtime_config(block)
        logger.info(f"[{session_info}] [{block['name']} ({block['block_type']})] Constructing")
        chain = build_runnable_from_block(
            block=block,
            session_info=session_info,
            extractors=extractors,
            api_key=api_key,
        )
        chain_parts.append(chain)
    # chain_parts = [build_runnable_from_block(b, debug_mode) for b in structure]

    # if len(chain_parts) == 1:
    #     return chain_parts[0]

    # if len(chain_parts) == 1:
    #     final_chain = chain_parts[0]
    # else:
    #     final_chain = RunnableSequence(*chain_parts)
    #
    # # Добавляем финальный шаг для сохранения inputs и результата
    # final_step = RunnableLambda(lambda x: {"inputs": x, "final_result": final_chain.invoke(x)})
    #
    # # Возвращаем цепочку с финальным шагом
    # return final_step
    if len(chain_parts) == 1:
        return chain_parts[0]

    # добавляем обёртку только в конец
    last_block = chain_parts.pop()
    chain_parts.append(with_passthrough(last_block, "final_result"))

    return RunnableSequence(*chain_parts)


def with_passthrough(last_chain, output_key: str):
    """Оборачивает последний блок так, чтобы вернуть inputs + результат"""

    def _fn(inputs: dict):
        # print("with_passthrough inputs ")
        # print_dict_structure(inputs)
        # print("\n")

        result = last_chain.invoke(inputs)
        outputs = {
            **inputs,
            output_key: result,
        }
        outputs = unpack_dict_keys(data=outputs, keys_to_unpack=["original_inputs", "report_and_router", "final_result"])
        # print("with_passthrough outputs ")
        # print_dict_structure(outputs)
        # print("\n")
        return outputs

    return RunnableLambda(_fn)


def unpack_dict_keys(data: Dict[str, Any], keys_to_unpack: List[str]) -> Dict[str, Any]:
    """
    Рекурсивная распаковка словарей по заданным ключам.

    Args:
        data: словарь с результатами цепочек
        keys_to_unpack: список ключей, которые нужно разворачивать, если под ними dict

    Returns:
        Новый словарь без потерь данных.
    """
    unpacked = dict(data)  # копия, чтобы не портить оригинал
    changed = True

    while changed:
        changed = False
        for key in list(unpacked.keys()):
            if key in keys_to_unpack:
                value = unpacked.pop(key)
                if isinstance(value, dict):
                    unpacked.update(value)
                    changed = True
                else:
                    # если не dict → оставляем ключ как есть
                    unpacked[key] = value
    return unpacked
