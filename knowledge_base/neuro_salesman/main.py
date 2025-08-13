from typing import Dict

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableParallel

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import Runnable

from neuro_salesman.roles_config import EXTRACTOR_ROLES

DEFAULT_LLM_MODEL = "gpt-4.1-nano"

# --- Загрузка ключей
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- LLM
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,
)

# --- Векторная база
vectordb = FAISS.load_local(
    'test_faiss_db_it_courses',
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

greeting_identification_system_prompt = """
Ты должен выявить приветствие в тексте клиента.
Если приветствия нет — верни пустую строку.
"""

# === 1. Выявление приветствия
greeting_prompt = PromptTemplate(
    input_variables=["client_text", "instructions"],
    template="""{instructions} Текст клиента: {client_text}
Ответ:
"""
)
greeting_identification_chain = (
    LLMChain(llm=llm, prompt=greeting_prompt, output_key="greeting")
    .bind(instruction=greeting_identification_system_prompt)
)

# === 2. Извлечение сущностей
extractor_prompt = PromptTemplate(
    input_variables=["system_prompt", "instructions", "analysis_text"],
    template="""
{system_prompt}

{instructions}

Текст для анализа:
{analysis_text}

Ответ:
"""
)
extractor_chain = LLMChain(llm=llm, prompt=extractor_prompt, output_key="entities")

# === 3. Выделение ключевых фраз
topic_phrase_prompt = PromptTemplate(
    input_variables=["system_prompt", "instructions", "user_history", "manager_history"],
    template="""
{system_prompt}

{instructions}

Текст:
{user_history}

{manager_history}

Ответ:
"""
)
topic_phrase_chain = LLMChain(llm=llm, prompt=topic_phrase_prompt, output_key="topic_phrase")

# === 4. Суммаризация диалога
summarize_prompt = PromptTemplate(
    input_variables=["dialog_history", "last_statements"],
    template="""
Ты супер корректор, умеешь выделять в диалогах всё самое важное.
Суммаризируй Диалог, ничего не придумывай.

История предыдущих сообщений:
{dialog_history}

Последние сообщения:
{last_statements}

Ответ:
"""
)
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt, output_key="summary")

# === 5. Роутинг к экспертам
router_prompt = PromptTemplate(
    input_variables=["system_prompt", "instructions", "question", "summary_history", "summary_exact"],
    template="""
{system_prompt}

{instructions}

Вопрос клиента:
{question}

Хронология предыдущих сообщений:
{summary_history}

Точное саммари:
{summary_exact}

Ответ:
"""
)
router_chain = LLMChain(llm=llm, prompt=router_prompt, output_key="experts_list")

# === 6. Финальная цепочка (SequentialChain)
full_chain = SequentialChain(
    chains=[
        greeting_chain,
        extractor_chain,
        topic_phrase_chain,
        summarize_chain,
        router_chain
    ],
    input_variables=["client_text", "system_prompt", "instructions", "analysis_text", "user_history", "manager_history",
                     "dialog_history", "last_statements", "question", "summary_history", "summary_exact"],
    output_variables=["greeting", "entities", "topic_phrase", "summary", "experts_list"],
    verbose=True
)

# --- Пример запуска
if __name__ == "__main__":
    result = full_chain.run(
        client_text="Здравствуйте! Хочу узнать про курс по Python.",
        system_prompt="Выяви сущности",
        instructions="Извлеки только нужные данные",
        analysis_text="Клиент: Хочу узнать про курс по Python",
        user_history="Хочу узнать про курс по Python",
        manager_history="",
        dialog_history="",
        last_statements="Хочу узнать про курс по Python",
        question="Хочу узнать про курс по Python",
        summary_history="",
        summary_exact=""
    )
    print(result)

# --- Карта экстракторов
extractors = {
    "needs": "needs_extractor",
    "benefits": "benefits_extractor",
    "objections": "objection_detector",
    "resolved_objections": "resolved_objection_detector",
    "tariffs": "tariff_extractor"
}

# --- Базовый шаблон для всех экстракторов
prompt_template = PromptTemplate(
    input_variables=["system_prompt", "history", "question"],
    template="""
{system_prompt}

История диалога:
{history}

Вопрос клиента:
{question}

Ответ:
"""
)


class VerboseLLMChain(Runnable):
    def __init__(self, chain: LLMChain, verbose: bool = False):
        self.chain = chain
        self.verbose = verbose

    def invoke(self, inputs: dict):
        if self.verbose:
            print("\n--- Input to LLM ---")
            print(inputs)
        result = self.chain.invoke(inputs)
        if self.verbose:
            print("--- Output from LLM ---")
            print(result)
        return result


def make_extractor_chain(
        chain_name: str,
        chain_config: Dict,
):
    """
    chain_config: словарь из EXTRACTOR_ROLES
    chain_name: ключ для результата
    """

    llm = ChatOpenAI(
        model=chain_config.get("model", DEFAULT_LLM_MODEL),
        temperature=chain_config.get("temperature", 0)
    )

    # Объединяем system_prompt + instructions в единый prompt с переменной {text}
    prompt_template = PromptTemplate(
        template="{system_prompt}\n{instructions}\n\nТекст для анализа:\n{text}\n\nОтвет:",
        input_variables=["system_prompt", "instructions", "text"]
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key=output_key
    )

    return chain


# --- Основная функция (упрощённая версия get_seller_answer)
def get_seller_answer(user_message: str, histories: Dict, verbose=False):
    # histories — словарь с историями
    # histories["manager"] — список сообщений менеджера
    # histories["user"] — список сообщений пользователя

    # Формируем входные данные
    inputs = {
        "history": "\n".join(histories.get("manager", [])),
        "question": user_message
    }

    # --- Формируем параллельный набор LLMChain
    chains = {}
    for extractor, extractor_config in extractors.items():
        name = extractor_config.get("name"),
        temp = extractor_config.get("temperature"),
        system_prompt = extractor_config.get("system_prompt"),
        instructions = extractor_config.get("instructions"),
        llm = ChatOpenAI(model=llm_cfg["model"], temperature=llm_cfg["temperature"])
        chains[extractor] = LLMChain(
            llm=llm,
            prompt=prompt_template,
            output_key=extractor
        ).bind(system_prompt=EXTRACTORS[extractor_key])

    parallel_extractors = RunnableParallel(**chains)



    # Запуск всех экстракторов параллельно
    extractors_data = parallel_extractors.invoke(inputs)

    if verbose:
        print("Результаты экстракции:", neuro_data)

    return neuro_data


# === Пример запуска ===
if __name__ == "__main__":
    histories = {
        "manager": ["Менеджер: Это отличный курс по Python."],
        "user": ["Клиент: Хочу узнать про курс Python."]
    }
    user_message = "А сколько стоит курс?"
    neuro_data = get_seller_answer(user_message, histories, verbose=True)
