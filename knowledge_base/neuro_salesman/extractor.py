import time
from copy import copy
from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, Runnable, RunnablePassthrough

from neuro_salesman.chains.chain_logger import ChainLogger
from neuro_salesman.chains.generic_runnable import GenericRunnable
from neuro_salesman.chains.keyed_runnable import KeyedRunnable
from neuro_salesman.config import DEFAULT_LLM_MODEL, EMPTY_MESSAGE, LLM_MAX_RETRIES, LLM_TIMEOUT
from neuro_salesman.utils import print_dict_structure, extract_list, merge_unique_lists


def make_extractor_chain(
        chain_name: str,
        chain_config: Dict[str, Any],
        session_info: str,
) -> "GenericRunnable":
    """
    Универсальный билдер цепочек-«экстракторов» для анализа текста.

    Экстрактор получает текст из поля `inputs[target]`, прогоняет через LLM
    с заранее заданным системным промптом и инструкциями, и возвращает
    результат под ключом `chain_name`.

    Args:
        chain_name (str):
            Имя цепочки (ключ для результата в выходном словаре).

        chain_config (Dict[str, Any]):
            Конфигурация экстрактора. Поддерживаемые ключи:
              - "model_name" (str): название модели (по умолчанию DEFAULT_LLM_MODEL).
              - "model_temperature" (float): температура генерации (по умолчанию 0).
              - "system_prompt" (str): системный промпт для LLM.
              - "instructions" (str): инструкции, добавляемые к промпту.
              - "verbose_name" (str): человеко-читаемое имя цепочки (для логов).
              - "target" (str): название поля во входных данных,
                из которого брать текст для анализа (**обязательный параметр**).

        session_info (str):
            Идентификатор или описание сессии (используется в логгере).

    Returns:
        GenericRunnable:
            Объект-обертка над LLM-цепочкой с маппингом входов и выходов.

    Raises:
        ValueError: если в конфиге отсутствует ключ "target".
    """
    # --- Логгер для отладки и мониторинга ---
    logger = ChainLogger(prefix=f"{chain_name} (extractor)")
    logger.log(session_info, "info", "Chain started")

    # --- Чтение параметров из конфигурации ---
    model_name = chain_config.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = chain_config.get("model_temperature", 0)
    system_prompt = chain_config.get("system_prompt", "")
    instructions = chain_config.get("instructions", "")
    verbose_name = chain_config.get("verbose_name", "Extractor")
    target = chain_config.get("target")

    if not target:
        raise ValueError(f"Extractor {chain_name} не имеет target в конфиге")

    # --- Инициализация LLM ---
    llm = ChatOpenAI(
        model=model_name,
        temperature=model_temperature,
        max_retries=LLM_MAX_RETRIES,
        timeout=LLM_TIMEOUT,
    )
    logger.log(
        session_info,
        "info",
        f"Chain creation started (model={model_name}, temperature={model_temperature})"
    )

    # --- Шаблон промпта ---
    # system_prompt + instructions → контекст
    # text → текст для анализа
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system_prompt}"),
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\nТекст для анализа:\n{text}\n\nОтвет:"
        )
    ])

    # Сам пайплайн: промпт → LLM
    chain = prompt_template | llm

    # --- Маппинг входов ---
    def input_mapping(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подготавливает словарь для промпта:
        - забирает текст из inputs[target],
        - добавляет system_prompt и instructions.
        """
        target = chain_config.get("target")

        text = inputs.get(target, "")
        if isinstance(text, AIMessage):
            text = text.content
        return {
            "system_prompt": system_prompt,
            "instructions": instructions,
            "text": text,
        }

    # --- Маппинг выходов ---
    def output_mapping(result: Any, inputs: Dict[str, Any]) -> Any:
        """
        Объединяет исходные inputs с результатом LLM.
        Результат сохраняется под ключом `chain_name`.
        """
        # print("EXTRACTOR ", chain_name)
        # print_dict_structure(copy(inputs))
        # print("\n")
        accumulate_history = chain_config.get("accumulate_history", False)
        output_format = chain_config.get("output_format", "str")
        print(f"{chain_name}: {accumulate_history=}")
        print(f"{chain_name}: {output_format=}")

        extractor_result = result.content if result and result.content else None

        if extractor_result is None:
            if output_format == "list":
                extractor_result = []
            else:
                extractor_result = ""
        else:
            if output_format == "list":
                extractor_result = extract_list(extractor_result)
            else:
                extractor_result = extractor_result.strip().strip('"')

        print(f"{chain_name}: {extractor_result=}")
        print(f"{chain_name}: {result=}")

        if accumulate_history:
            extractor_history = inputs.get(f"{chain_name}_history")
            print(f"{chain_name}: {extractor_history=}")
            if output_format == "list":
                if extractor_history:
                    output_extractor_result = merge_unique_lists(extractor_history, extractor_result)
                else:
                    output_extractor_result = extractor_result
            else:
                if extractor_history:
                    output_extractor_result = f"{extractor_history}. {extractor_result}".strip(". ")
                else:
                    output_extractor_result = extractor_result
        else:
            output_extractor_result = extractor_result

        print(f"{chain_name}: {output_extractor_result=}")
        inputs.update({f"{chain_name}_history": output_extractor_result})
        return result

    # --- Возврат готовой обёртки ---
    return GenericRunnable(
        chain=chain,
        output_key=chain_name,
        prefix=verbose_name,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
    )


def build_parallel_extractors(extractors: dict, debug_mode: bool = False):
    """
    Создает параллельный runnable, который запускает все экстракторы одновременно.

    Пошагово:
    1. Для каждого экстрактора из EXTRACTOR_ROLES создается BoundChainRunnable через make_extractor_chain.
    2. Добавляется RunnablePassthrough под ключ 'original_inputs' для пропуска исходных данных.
    3. Возвращается RunnableParallel с набором цепочек.

    Args:
        debug_mode (bool): Включает подробный вывод в консоль и debug-лог.

    Returns:
        RunnableParallel: параллельный runnable с цепочками-экстракторами и passthrough для оригинальных inputs.
    """
    chains = {}
    for extractor, extractor_config in extractors.items():
        chains[extractor] = make_extractor_chain(extractor_config, debug_mode=debug_mode)
    chains['original_inputs'] = RunnablePassthrough()
    return RunnableParallel(**chains)
