from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from neuro_salesman.config import DEFAULT_LLM_MODEL

greeting_identification_system_prompt = """
Ты должен выявить приветствие в тексте клиента.
Если приветствия нет — верни пустую строку.
"""

# Создание цепочки для выявления приветствия
def create_extract_greeting_chain(debug_mode: bool = False):
    llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0)

    # Промпт с фиксированным instructions
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(greeting_identification_system_prompt),
        HumanMessagePromptTemplate.from_template("Текст для анализа:\n{last_message_from_client}\n\nОтвет:")
    ])

    # Создаём цепочку
    chain = prompt_template | llm

    # Оборачиваем результат в словарь с ключом 'greeting'
    class KeyedRunnable(Runnable):
        def __init__(self, chain, output_key):
            self.chain = chain
            self.output_key = output_key

        def invoke(self, inputs, config=None, **kwargs):
            text = inputs.get("last_message_from_client", "")
            if not text or not text.strip():
                if debug_mode:
                    print("[Greeting Extractor] Текст пуст — модель не вызывалась.")
                return {**inputs, self.output_key: ""}
            result = self.chain.invoke(inputs, config=config, **kwargs)
            if debug_mode:
                print(f"[Greeting Extractor] Input: {inputs}")
                print(f"[Greeting Extractor] Output: {{{self.output_key}: {result}}}")
            return {**inputs, self.output_key: result}

    return KeyedRunnable(chain, "greeting")


def create_remove_greeting_chain(debug_mode: bool = False):
    """
    Создает цепочку для удаления приветствий из текста в стиле LangChain.
    """
    llm = ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        temperature=0
    )

    system_prompt = """
    Ты — высокоточный редактор текста.
    Твоя задача: удалить из начала текста только приветствие (вежливую фразу начала общения).
    Не изменяй остальной текст.
    Не добавляй пояснений, комментариев или новых слов.
    Возвращай только отредактированный текст, без лишнего форматирования.
    """

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(
            "Исходный текст: Добрый день, Кира, я готов рассказать Вам о курсе подробнее. Начнем с тарифов?\n\nОтвет:"
        ),
        AIMessage(content="Кира, я готов рассказать Вам о курсе подробнее. Начнем с тарифов?"),
        HumanMessagePromptTemplate.from_template(
            "Исходный текст: Привет, Сергей! Сегодня у нас отличная погода.\n\nОтвет:"
        ),
        AIMessage(content="Сергей! Сегодня у нас отличная погода."),
        HumanMessagePromptTemplate.from_template(
            "Исходный текст: Здравствуйте, коллеги. Начнем собрание.\n\nОтвет:"
        ),
        AIMessage(content="Коллеги. Начнем собрание."),
        HumanMessagePromptTemplate.from_template(
            "Исходный текст: {text}\n\nОтвет:"
        )
    ])

    class KeyedRunnable(Runnable):
        def __init__(self, chain, output_key, debug_mode):
            self.chain = chain
            self.output_key = output_key
            self.debug_mode = debug_mode

        def invoke(self, inputs, config=None, **kwargs):
            text = inputs.get("stylized_answer", "")
            if not text or not text.strip():
                if self.debug_mode:
                    print("[Debug Remove Greeting Chain] Текст пуст — модель не вызывалась.")
                return {**inputs, self.output_key: ""}

            if self.debug_mode:
                print(f"[Debug Remove Greeting Chain] inputs: {inputs}")
                print(f"[Debug Remove Greeting Chain] text: {text}")

            try:
                result = self.chain.invoke(
                    {"text": text},
                    config=config,
                    **kwargs
                )
                if self.debug_mode:
                    print(f"[Debug Remove Greeting Chain] Output: {{{self.output_key}: {result}}}")
                return {**inputs, self.output_key: result.content.strip()}
            except Exception as e:
                # logger.error(f"Ошибка в Remove Greeting Chain: {str(e)}")
                if self.debug_mode:
                    print(f"[Debug Remove Greeting Chain] Ошибка: {str(e)}")
                return {**inputs, self.output_key: text}

    chain = prompt_template | llm
    return KeyedRunnable(chain, "answer_without_greetings", debug_mode)


