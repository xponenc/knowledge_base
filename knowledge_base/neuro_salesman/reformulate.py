from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

from neuro_salesman.chains.chain_logger import ChainLogger
from neuro_salesman.chains.generic_runnable import GenericRunnable
from neuro_salesman.config import DEFAULT_LLM_MODEL, LLM_MAX_RETRIES, LLM_TIMEOUT


def make_reformulator_chain(
        chain_name: str,
        chain_config: Dict[str, Any],
        session_info: str,
) -> GenericRunnable:
    """
    Возвращает переформулированный вопрос (если нужно) и флаг, указывающий на изменение.
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
            "{instructions}\n\nChat history: {chat_history}:\nCurrent Questions:{question}\n\nОтвет:"
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
        return {
            "system_prompt": system_prompt,
            "instructions": instructions,
            "chat_history": "\n".join(inputs.get("histories", [])),
            "question": inputs.get("last message from client", ""),
        }

    # --- Маппинг выходов ---
    def output_mapping(result: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Объединяет исходные inputs с результатом LLM.
        Результат сохраняется под ключом `chain_name`.
        """
        return result

    # --- Возврат готовой обёртки ---
    return GenericRunnable(
        chain=chain,
        output_key=chain_name,
        prefix=verbose_name,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
    )
