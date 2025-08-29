from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from neuro_salesman.chains.chain_logger import ChainLogger
from neuro_salesman.chains.generic_runnable import GenericRunnable
from neuro_salesman.config import DEFAULT_LLM_MODEL, LLM_MAX_RETRIES, LLM_TIMEOUT, EMPTY_MESSAGE
from neuro_salesman.roles_config import NEURO_SALER
from neuro_salesman.utils import print_dict_structure


# def create_stylist_chain(
#     chain_name: str,
#     chain_config: Dict[str, Any],
#     session_info: str,
# ) -> GenericRunnable:
#     """
#     Создает цепочку для стилизации ответа в стиле LangChain.
#
#     Args:
#         chain_name (str): Имя цепочки (для логирования).
#         chain_config (Dict[str, Any]): Конфигурация модели и промптов.
#         session_info (str): Информация о сессии (для логгирования).
#         debug_mode (bool): Включить отладочный вывод.
#
#     Returns:
#         GenericRunnable: Обертка над LLM-цепочкой для стилизации.
#     """
#
#     logger = ChainLogger(prefix=f"{chain_name} (stylist)")
#     logger.log(session_info, "info", "Chain started")
#
#     # --- Конфигурация ---
#     model_name = chain_config.get("model_name", DEFAULT_LLM_MODEL)
#     model_temperature = chain_config.get("model_temperature", 0)
#     system_prompt = chain_config.get("system_prompt", "")
#     instructions = chain_config.get("instructions", "")
#
#     # --- Инициализация LLM ---
#     llm = ChatOpenAI(
#         model=model_name,
#         temperature=model_temperature,
#         max_retries=LLM_MAX_RETRIES,
#         timeout=LLM_TIMEOUT,
#     )
#
#     # --- Prompt-шаблон ---
#     prompt_template = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(system_prompt),
#         HumanMessagePromptTemplate.from_template(
#             "{instructions}\n\n"
#             "Исходный текст: {answers_content}\n\n"
#             "Ответ:"
#         ),
#         # few-shot примеры
#         HumanMessagePromptTemplate.from_template(
#             "{instructions}\n\n"
#             "Исходный текст: Кира, я рад, что ты заинтересовалась нашими курсами. "
#             "Наши программы обучения позволят тебе погрузиться в мир искусственного интеллекта..."
#             "\n\nОтвет:"
#         ),
#         AIMessage(content='''
#             Кира, я рад, что Вы заинтересовались нашими курсами. Наши образовательные программы позволят
#             Вам окунуться в мир искусственного интеллекта с самого начала...
#         '''),
#         HumanMessagePromptTemplate.from_template(
#             "{instructions}\n\n"
#             "Исходный текст: У нас в АДО самая обширная база учебного контента..."
#             "\n\nОтвет:"
#         ),
#         AIMessage(content='''
#             У нас в УИИ самая обширная база учебного контента по искусственному интеллекту...
#         '''),
#         HumanMessagePromptTemplate.from_template(
#             "{instructions}\n\n"
#             "Исходный текст: {answers_content}\n\n"
#             "Ответ:"
#         )
#     ])
#
#     chain = prompt_template | llm
#
#     # --- input_mapping ---
#     def input_mapping(inputs: Dict[str, Any]) -> Dict[str, Any]:
#         senior_answer = inputs.get("senior", EMPTY_MESSAGE)
#         answers_content = senior_answer.content
#
#         return {"instructions": instructions, "answers_content": answers_content}
#
#     # --- output_mapping ---
#     def output_mapping(result: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
#         output = {**inputs, "stylized_answer": result}
#         # print("STYLIST ", chain_name)
#         # print_dict_structure(output)
#         # print("\n")
#         return output
#
#     # --- Обертка ---
#     return GenericRunnable(
#         chain=chain,
#         output_key="stylized_answer",
#         prefix=f"{chain_name} (stylist)",
#         input_mapping=input_mapping,
#         output_mapping=output_mapping,
#     )


def create_stylist_chain(
    chain_name: str,
    chain_config: Dict[str, Any],
    session_info: str,
) -> GenericRunnable:
    """
    Создает цепочку для стилизации ответа в стиле LangChain.

    Цель цепочки: привести ответ модели к нужному стилю,
    сохраняя смысл и корректные наименования (например, академия, курсы),
    избегая подмен из примеров few-shot.

    Args:
        chain_name (str): Имя цепочки (для логирования).
        chain_config (Dict[str, Any]): Конфигурация модели и промптов.
            Ключи:
              - model_name: str, модель для LLM (по умолчанию DEFAULT_LLM_MODEL)
              - model_temperature: float, температура генерации
              - system_prompt: str, системный промпт для LLM
              - instructions: str, инструкции для стилизации текста
        session_info (str): Информация о сессии (для логгирования).
        debug_mode (bool): Включить отладочный вывод.

    Returns:
        GenericRunnable: Обертка над LLM-цепочкой для стилизации.
    """

    # --- Логгер ---
    logger = ChainLogger(prefix=f"{chain_name} (stylist)")
    logger.log(session_info, "info", "Chain started")

    # --- Чтение конфигурации ---
    model_name = chain_config.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = chain_config.get("model_temperature", 0)
    system_prompt = chain_config.get("system_prompt", "")
    instructions = chain_config.get("instructions", "")

    # --- Инициализация LLM ---
    llm = ChatOpenAI(
        model=model_name,
        temperature=model_temperature,
        max_retries=LLM_MAX_RETRIES,
        timeout=LLM_TIMEOUT,
    )

    # --- Prompt-шаблон ---
    # Объединяем системный промпт, инструкции и few-shot примеры
    # Внимание: примеры нейтральные, без реальных аббревиатур/имен организаций
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),

        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Исходный текст: {answers_content}\n\n"
            "Ответ:"
        ),

        # few-shot примеры (нейтральные)
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Исходный текст: Кандидат проявил интерес к курсам. "
            "Организация X предлагает программы обучения искусственному интеллекту..."
            "\n\nОтвет:"
        ),
        AIMessage(content='''
            Кандидат проявил интерес к курсам. Программы Организации X позволят
            изучить искусственный интеллект и программирование с практической стороны.
        '''),

        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Исходный текст: Организация X обладает богатым учебным контентом по ИИ..."
            "\n\nОтвет:"
        ),
        AIMessage(content='''
            Организация X предлагает обширный учебный контент по искусственному интеллекту
            и программированию, позволяя эффективно осваивать новые навыки.
        '''),

        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Исходный текст: {answers_content}\n\n"
            "Ответ:"
        )
    ])

    # --- Цепочка: промпт → LLM ---
    chain = prompt_template | llm

    # --- input_mapping ---
    def input_mapping(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подготавливает данные для промпта.
        Берем senior ответ (или пустой) и передаем в промпт.
        """
        senior_answer = inputs.get("senior", EMPTY_MESSAGE)
        answers_content = senior_answer.content
        return {"instructions": instructions, "answers_content": answers_content}

    # --- output_mapping ---
    def output_mapping(result: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Возвращает словарь с результатом стилизации.
        """
        output = {**inputs, "stylized_answer": result}
        return output

    # --- Возврат обертки ---
    return GenericRunnable(
        chain=chain,
        output_key="stylized_answer",
        prefix=f"{chain_name} (stylist)",
        input_mapping=input_mapping,
        output_mapping=output_mapping,
    )
