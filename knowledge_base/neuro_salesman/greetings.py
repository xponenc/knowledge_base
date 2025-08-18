import time

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from neuro_salesman.chains.keyed_runnable import KeyedRunnable
from neuro_salesman.config import DEFAULT_LLM_MODEL, EMPTY_MESSAGE, LLM_TIMEOUT, LLM_MAX_RETRIES
from neuro_salesman.roles_config import GREETING_EXTRACTOR, REMOVE_GREETING_CONFIG
from utils.setup_logger import setup_logger

logger = setup_logger(name=__file__, log_dir="logs/neuro_salesman", log_file="ns.log")


def create_extract_greeting_chain(debug_mode: bool = False):
    """
    Создает цепочку для выявления приветствия в последнем сообщении клиента.

    Пошагово:
    1. Создает LLM-модель с параметрами из GREETING_EXTRACTOR.
    2. Формирует промпт из system_prompt и instructions.
    3. Оборачивает цепочку в KeyedRunnable, которая:
        - проверяет, пустой ли текст;
        - вызывает модель;
        - добавляет результат под ключ "greeting" в inputs;
        - логгирует вход и выход в зависимости от debug_mode.

    Args:
        debug_mode (bool): Включает подробный вывод в консоль и DEBUG-логгер.

    Returns:
        KeyedRunnable: объект Runnable, который при вызове invoke возвращает inputs с добавленным ключом 'greeting'.
    """

    model_name = GREETING_EXTRACTOR.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = GREETING_EXTRACTOR.get("model_temperature", 0)
    system_prompt = GREETING_EXTRACTOR.get("system_prompt", "")
    instructions = GREETING_EXTRACTOR.get("instructions", "")
    verbose_name = GREETING_EXTRACTOR.get("verbose_name", "Greeting Extractor")
    logger.info(f"[{verbose_name}] chain started")

    # Создаем LLM-модель
    llm = ChatOpenAI(model=model_name,
                     temperature=model_temperature,
                     max_retries=LLM_MAX_RETRIES,
                     timeout=LLM_TIMEOUT)

    # Промпт с фиксированными instructions
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(
            f"{instructions}:\n{{last message from client}}\n\nОтвет:"
        )
    ])

    # Цепочка LLM + промпт
    chain = prompt_template | llm
    logger.info(f"[{verbose_name}] chain created (model={model_name}, temperature={model_temperature})")

    # class KeyedRunnable(Runnable):
    #     """
    #     Обертка для цепочки, возвращающая результат под ключом output_key.
    #
    #     Attributes:
    #         chain (Runnable): цепочка, которую нужно вызвать.
    #         output_key (str): ключ, под которым сохранять результат в словаре inputs.
    #     """
    #
    #     def _log(self, session_info, level, message):
    #         if self.debug_mode:
    #             print(f"[Greeting Extractor][{session_info}] {message}")
    #         getattr(logger, level)(f"[Greeting Extractor][{session_info}] {message}")
    #
    #     def __init__(self, chain, output_key, debug_mode):
    #         self.chain = chain
    #         self.output_key = output_key
    #         self.debug_mode = debug_mode
    #
    #     def invoke(self, inputs, config=None, **kwargs):
    #         """
    #         Вызывает цепочку на последнем сообщении клиента и возвращает обновленный словарь inputs.
    #
    #         Args:
    #             inputs (dict): словарь с входными данными, ожидается ключ "last message from client".
    #             config: опциональная конфигурация вызова.
    #             **kwargs: дополнительные параметры вызова.
    #
    #         Returns:
    #             dict: исходные inputs с добавленным ключом output_key и результатом цепочки.
    #         """
    #         start_time = time.monotonic()
    #         session_info = f"[{inputs.get('session_type', 'n/a')}:{inputs.get('session_id', 'n/a')}]"
    #         text = inputs.get("last message from client", "")
    #         if not text or not text.strip():
    #             self._log(session_info, "warning", "пустой текст, модель не вызвалась")
    #             if self.debug_mode:
    #                 print(f"[Greeting Extractor][{session_info}]: пустой текст, модель не вызвалась")
    #             return {**inputs, self.output_key: EMPTY_MESSAGE}
    #
    #         self._log(session_info, "info", "started")
    #
    #         try:
    #             result = self.chain.invoke(inputs,
    #                                        config=config,
    #                                        **kwargs)
    #
    #             elapsed = time.monotonic() - start_time
    #             self._log(session_info, "debug", f"input: {inputs}")
    #             self._log(session_info, "info", f"finished in {elapsed:.2f}s")
    #             self._log(session_info, "debug", f"output: {{{self.output_key}: {result}}}")
    #             return {**inputs, self.output_key: result}
    #
    #         except Exception as e:
    #             logger.error(f"[Greeting Extractor][{session_info}] Ошибка: {str(e)}", exc_info=True)
    #             if debug_mode:
    #                 print(f"[Greeting Extractor][{session_info}] Ошибка: {str(e)}")
    #             return {**inputs, self.output_key: EMPTY_MESSAGE}

    return KeyedRunnable(chain, prefix=verbose_name, output_key="greeting", debug_mode=debug_mode)


# def create_remove_greeting_chain(debug_mode: bool = False):
#     """
#     Создает цепочку для удаления приветствий из текста в стиле LangChain.
#     """
#     llm = ChatOpenAI(
#         model=DEFAULT_LLM_MODEL,
#         temperature=0
#     )
#
#     system_prompt = """
#     Ты — высокоточный редактор текста.
#     Твоя задача: удалить из начала текста только приветствие (вежливую фразу начала общения).
#     Не изменяй остальной текст.
#     Не добавляй пояснений, комментариев или новых слов.
#     Возвращай только отредактированный текст, без лишнего форматирования.
#     """
#
#     prompt_template = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(system_prompt),
#         HumanMessagePromptTemplate.from_template(
#             "Исходный текст: Добрый день, Кира, я готов рассказать Вам о курсе подробнее. Начнем с тарифов?\n\nОтвет:"
#         ),
#         AIMessage(content="Кира, я готов рассказать Вам о курсе подробнее. Начнем с тарифов?"),
#         HumanMessagePromptTemplate.from_template(
#             "Исходный текст: Привет, Сергей! Сегодня у нас отличная погода.\n\nОтвет:"
#         ),
#         AIMessage(content="Сергей! Сегодня у нас отличная погода."),
#         HumanMessagePromptTemplate.from_template(
#             "Исходный текст: Здравствуйте, коллеги. Начнем собрание.\n\nОтвет:"
#         ),
#         AIMessage(content="Коллеги. Начнем собрание."),
#         HumanMessagePromptTemplate.from_template(
#             "Исходный текст: {text}\n\nОтвет:"
#         )
#     ])
#
#     class KeyedRunnable(Runnable):
#         def __init__(self, chain, output_key, debug_mode):
#             self.chain = chain
#             self.output_key = output_key
#             self.debug_mode = debug_mode
#
#         def invoke(self, inputs, config=None, **kwargs):
#             text = inputs.get("stylized_answer", "")
#             if not text or not text.strip():
#                 if self.debug_mode:
#                     print("[Debug Remove Greeting Chain] Текст пуст — модель не вызывалась.")
#                 return {**inputs, self.output_key: ""}
#
#             if self.debug_mode:
#                 print(f"[Debug Remove Greeting Chain] inputs: {inputs}")
#                 print(f"[Debug Remove Greeting Chain] text: {text}")
#
#             try:
#                 result = self.chain.invoke(
#                     {"text": text},
#                     config=config,
#                     **kwargs
#                 )
#                 if self.debug_mode:
#                     print(f"[Debug Remove Greeting Chain] Output: {{{self.output_key}: {result}}}")
#                 return {**inputs, self.output_key: result.content.strip()}
#             except Exception as e:
#                 # logger.error(f"Ошибка в Remove Greeting Chain: {str(e)}")
#                 if self.debug_mode:
#                     print(f"[Debug Remove Greeting Chain] Ошибка: {str(e)}")
#                 return {**inputs, self.output_key: text}
#
#     chain = prompt_template | llm
#     return KeyedRunnable(chain, "answer_without_greetings", debug_mode)

#
#
# def create_remove_greeting_chain(debug_mode: bool = False):
#     """
#     Создает цепочку для удаления приветствий из текста.
#     Возвращает Runnable, который всегда возвращает AIMessage,
#     чтобы сохранить единообразие типов данных.
#     """
#
#     model_name = REMOVE_GREETING_CONFIG.get("model_name", DEFAULT_LLM_MODEL)
#     model_temperature = REMOVE_GREETING_CONFIG.get("model_temperature", 0)
#     system_prompt = REMOVE_GREETING_CONFIG.get("system_prompt", "")
#     instructions = REMOVE_GREETING_CONFIG.get("instructions", "")
#     examples = REMOVE_GREETING_CONFIG.get("instructions", [])
#
#     llm = ChatOpenAI(
#         model=model_name,
#         temperature=model_temperature
#     )
#
#     # Формируем prompt с примерами
#     messages = [
#         SystemMessagePromptTemplate.from_template(system_prompt)
#     ]
#     for example in examples:
#         messages.append(HumanMessagePromptTemplate.from_template(f"Исходный текст: {example['input']}\n\nОтвет:"))
#         messages.append(AIMessage(content=example["output"]))
#
#     # Шаблон для основного текста
#     messages.append(HumanMessagePromptTemplate.from_template("Исходный текст: {text}\n\nОтвет:"))
#
#     prompt_template = ChatPromptTemplate.from_messages(messages)
#     chain = prompt_template | llm
#
#     class KeyedRunnable(Runnable):
#         def __init__(self, chain, output_key, debug_mode):
#             self.chain = chain
#             self.output_key = output_key
#             self.debug_mode = debug_mode
#
#         def invoke(self, inputs, config=None, **kwargs):
#             text = inputs.get(REMOVE_GREETING_CONFIG["target"], "")
#             if not text or not text.strip():
#                 logger.warning("[Remove Greeting Chain] Пустой текст — модель не вызывалась")
#                 if self.debug_mode:
#                     print("[Debug Remove Greeting Chain] Текст пуст — модель не вызывалась.")
#                 return {**inputs, self.output_key: EMPTY_MESSAGE}
#
#             if self.debug_mode:
#                 print(f"[Debug Remove Greeting Chain] input text: {text}")
#
#             try:
#                 result = self.chain.invoke({"text": text}, config=config, **kwargs)
#                 if self.debug_mode:
#                     print(f"[Debug Remove Greeting Chain] Output: {result.content}")
#                 return {**inputs, self.output_key: result}
#             except Exception as e:
#                 logger.error(f"[Remove Greeting Chain] Ошибка: {str(e)}")
#                 if self.debug_mode:
#                     print(f"[Debug Remove Greeting Chain] Ошибка: {str(e)}")
#                 return {**inputs, self.output_key: EMPTY_MESSAGE}
#
#     return KeyedRunnable(chain, "answer_without_greetings", debug_mode)


def create_remove_greeting_chain(debug_mode: bool = False):
    """
    Создает цепочку для удаления приветствий из текста.
    Возвращает KeyedRunnable, который берет текст из REMOVE_GREETING_CONFIG['target']
    и возвращает результат под ключом 'answer_without_greetings'.
    """

    model_name = REMOVE_GREETING_CONFIG.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = REMOVE_GREETING_CONFIG.get("model_temperature", 0)
    system_prompt = REMOVE_GREETING_CONFIG.get("system_prompt", "")
    instructions = REMOVE_GREETING_CONFIG.get("instructions", "")
    examples = REMOVE_GREETING_CONFIG.get("examples", [])
    verbose_name = REMOVE_GREETING_CONFIG.get("verbose_name", "Remove Greeting Chain")
    target_key = REMOVE_GREETING_CONFIG.get("target", "")

    logger.info(f"[{verbose_name}] chain started")

    llm = ChatOpenAI(
        model=model_name,
        temperature=model_temperature,
        max_retries=LLM_MAX_RETRIES,
        timeout=LLM_TIMEOUT
    )

    # Формируем prompt с примерами
    messages = [SystemMessagePromptTemplate.from_template(system_prompt)]
    for example in examples:
        messages.append(HumanMessagePromptTemplate.from_template(f"Исходный текст: {example['input']}\n\nОтвет:"))
        messages.append(AIMessage(content=example["output"]))
    messages.append(HumanMessagePromptTemplate.from_template(f"{instructions}:\n{{text}}\n\nОтвет:"))

    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = prompt_template | llm

    class RemoveGreetingRunnable(KeyedRunnable):
        """
        Специализированный KeyedRunnable для удаления приветствий.
        Берет текст из inputs[target_key] и возвращает результат под ключом output_key.
        """
        def invoke(self, inputs, config=None, **kwargs):
            text = inputs.get(target_key, "")
            return super().invoke({"text": text}, config=config, **kwargs)

    return RemoveGreetingRunnable(chain, output_key="answer_without_greetings", prefix=verbose_name, debug_mode=debug_mode)