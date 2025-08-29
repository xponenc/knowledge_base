from pprint import pprint
from typing import Dict, Any, List

from langchain.memory import ConversationSummaryMemory
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from neuro_salesman.chains.chain_logger import ChainLogger
from neuro_salesman.chains.generic_runnable import GenericRunnable
from neuro_salesman.config import DEFAULT_LLM_MODEL, EMPTY_MESSAGE
from neuro_salesman.utils import print_dict_structure


def extract_list(inputs: Dict[str, Any], key: str) -> List[str]:
    """
    Извлекает список из AIMessage, проходя по вложенным словарям/спискам по ключу, пока не дойдёт до AIMessage.
    """
    msg = inputs.get(key)
    # Спускаемся по вложенным словарям/спискам, пока не дойдём до AIMessage
    while msg and not isinstance(msg, AIMessage):
        if isinstance(msg, dict):
            msg = msg.get(key)
        elif isinstance(msg, list):
            # Берём первый элемент списка
            msg = msg[0] if msg else None
        else:
            # Не словарь, не список, и не AIMessage
            msg = None

    # Теперь msg либо AIMessage, либо None
    if isinstance(msg, AIMessage) and msg.content:
        return [x.strip() for x in msg.content.split(",") if x.strip()]

    return []


def create_extractors_report(
        chain_name: str,
        session_info: str,
        extractors: Dict
) -> RunnableLambda:
    """
    Создает Runnable для генерации summary_exact на основе результатов экстракторов.

    Args:
        debug_mode (bool): включить логирование в консоль

    Returns:
        RunnableLambda: объект Runnable, который при вызове с inputs возвращает inputs + "summary_exact"
    """

    def _summary(inputs: Dict) -> Dict:
        session_info = f"{inputs.get('session_type', 'n/a')}:{inputs.get('session_id', 'n/a')}"
        logger = ChainLogger(prefix=f"{chain_name} (summary)")
        logger.log(session_info, "info", "Chain started")

        logger.log(session_info, "debug", f"inputs: {inputs}")

        summary_exact = []
        for extractor_name, extractor_verbose_name in extractors.items():
            extractor_output = extract_list(inputs, extractor_name)

            summary_exact.append(
                f"# {extractor_verbose_name}: {', '.join(extractor_output) if extractor_output else 'не обнаружено'}\n")
        # needs = extract_list(inputs, "needs"),
        # benefits = extract_list(inputs, "benefits")
        # objections = extract_list(inputs, "objections")
        # resolved_objections = extract_list(inputs, "resolved_objections")
        # tariffs = extract_list(inputs, "tariffs")
        #
        # summary_exact = (
        #     f"# 1. Выявлены Потребности: {', '.join(needs) if needs else 'потребностей не обнаружено'}\n"
        #     f"# 2. Рассказанные Преимущества: {', '.join(benefits) if benefits else 'преимущества не были рассказаны'}\n"
        #     f"# 3. Возражения клиента: {', '.join(objections) if objections else 'возражений не обнаружено'}\n"
        #     f"# 4. Возражения клиента отработаны: {', '.join(resolved_objections) if resolved_objections else 'отработки не обнаружено'}\n"
        #     f"# 5. Конкретика - оговоренная конкретика - курсы, цены: {', '.join(tariffs) if tariffs else 'не обнаружено'}\n"
        # )

        logger.log(session_info, "info", f"Generated extractors report:\n{summary_exact}")

        return {**inputs, "extractors_report": summary_exact}

    return RunnableLambda(_summary)


def update_session_summary(
        chain_name: str,
        session_info: str,
        chain_config: Dict
) -> RunnableLambda:
    """
            Обновляет session_summary с помощью ConversationSummaryMemory.

            Args:


            Returns:
                str: Обновлённое суммари для передачи в inputs.
            """

    def _update(inputs: Dict) -> str:
        """
        Обновляет session_summary с помощью ConversationSummaryMemory.

        Args:

            inputs (Dict[str, Any]): Словарь входных данных, может содержать
                                     текущее session_summary.
        Returns:
            str: Обновлённое суммари для передачи в inputs.
        """
        # --- Логгер для отладки и мониторинга ---
        logger = ChainLogger(prefix=f"{chain_name} (summary)")
        logger.log(session_info, "info", "Chain started")

        # --- Чтение параметров из конфигурации ---
        model_name = chain_config.get("model_name", DEFAULT_LLM_MODEL)
        model_temperature = chain_config.get("model_temperature", 0.2)
        max_token_limit = chain_config.get("max_token_limit", 800)
        verbose_name = chain_config.get("verbose_name", "Extractor")

        # Берём текущее суммари из inputs
        current_summary = inputs.get("current_session_summary", "")
        user_message = inputs.get("last message from client", "").lstrip("Клиент:").strip()
        assistant_reply = inputs.get("stylized_answer", EMPTY_MESSAGE).content

        # Инициализируем memory с LLM
        llm = ChatOpenAI(model=model_name, temperature=model_temperature)
        summary_memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="session_summary",
            max_token_limit=max_token_limit,
            return_messages=False,
            initial_summary=current_summary  # стартуем с существующего суммари
        )

        # Обновляем память последней парой сообщений
        summary_memory.save_context(
            {"user_message": user_message},
            {"assistant_reply": assistant_reply}
        )

        # Берём актуальное суммари и возвращаем
        updated_summary = summary_memory.load_memory_variables({}).get("session_summary", "")
        if isinstance(updated_summary, str):
            conversation_string_from_history = updated_summary
        else:
            # Если история - это список сообщений, то формируем строку из их содержимого
            conversation_string_from_history = "\n".join(message.content for message in updated_summary)
        return conversation_string_from_history

    return RunnableLambda(_update)
