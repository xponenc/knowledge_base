from typing import Dict, List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
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

from neuro_salesman.config import DEFAULT_LLM_MODEL
from neuro_salesman.expert import process_experts
from neuro_salesman.extractor import build_parallel_extractors
from neuro_salesman.roles_config import EXTRACTOR_ROLES
from neuro_salesman.router import create_router_chain
from neuro_salesman.summary import create_summary_exact

# --- Загрузка ключей
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# # --- LLM
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,
)

# # --- Векторная база
# vectordb = FAISS.load_local(
#     'test_faiss_db_it_courses',
#     OpenAIEmbeddings(),
#     allow_dangerous_deserialization=True
# )
#
greeting_identification_system_prompt = """
Ты должен выявить приветствие в тексте клиента.
Если приветствия нет — верни пустую строку.
"""

# # === 1. Выявление приветствия

# greeting_identification_chain = (
#     LLMChain(llm=llm, prompt=greeting_prompt, output_key="greeting")
#     .bind(instructions=greeting_identification_system_prompt)
# )

# Создание цепочки для выявления приветствия
def create_greeting_chain(debug_mode: bool = False):
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

# # === 2. Извлечение сущностей
# extractor_prompt = PromptTemplate(
#     input_variables=["system_prompt", "instructions", "analysis_text"],
#     template="""
# {system_prompt}
#
# {instructions}
#
# Текст для анализа:
# {analysis_text}
#
# Ответ:
# """
# )
# extractor_chain = LLMChain(llm=llm, prompt=extractor_prompt, output_key="entities")
#
# # === 3. Выделение ключевых фраз
# topic_phrase_prompt = PromptTemplate(
#     input_variables=["system_prompt", "instructions", "user_history", "manager_history"],
#     template="""
# {system_prompt}
#
# {instructions}
#
# Текст:
# {user_history}
#
# {manager_history}
#
# Ответ:
# """
# )
# topic_phrase_chain = LLMChain(llm=llm, prompt=topic_phrase_prompt, output_key="topic_phrase")
#
# # === 4. Суммаризация диалога
# summarize_prompt = PromptTemplate(
#     input_variables=["dialog_history", "last_statements"],
#     template="""
# Ты супер корректор, умеешь выделять в диалогах всё самое важное.
# Суммаризируй Диалог, ничего не придумывай.
#
# История предыдущих сообщений:
# {dialog_history}
#
# Последние сообщения:
# {last_statements}
#
# Ответ:
# """
# )
# summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt, output_key="summary")
#
# # === 5. Роутинг к экспертам
# router_prompt = PromptTemplate(
#     input_variables=["system_prompt", "instructions", "question", "summary_history", "summary_exact"],
#     template="""
# {system_prompt}
#
# {instructions}
#
# Вопрос клиента:
# {question}
#
# Хронология предыдущих сообщений:
# {summary_history}
#
# Точное саммари:
# {summary_exact}
#
# Ответ:
# """
# )
# router_chain = LLMChain(llm=llm, prompt=router_prompt, output_key="experts_list")

# === 6. Финальная цепочка (SequentialChain)
# full_chain = SequentialChain(
#     chains=[
#         greeting_chain,
#         extractor_chain,
#         topic_phrase_chain,
#         summarize_chain,
#         router_chain
#     ],
#     input_variables=["client_text", "system_prompt", "instructions", "analysis_text", "user_history", "manager_history",
#                      "dialog_history", "last_statements", "question", "summary_history", "summary_exact"],
#     output_variables=["greeting", "entities", "topic_phrase", "summary", "experts_list"],
#     verbose=True
# )

# # --- Пример запуска
# if __name__ == "__main__":
#     result = full_chain.run(
#         client_text="Здравствуйте! Хочу узнать про курс по Python.",
#         system_prompt="Выяви сущности",
#         instructions="Извлеки только нужные данные",
#         analysis_text="Клиент: Хочу узнать про курс по Python",
#         user_history="Хочу узнать про курс по Python",
#         manager_history="",
#         dialog_history="",
#         last_statements="Хочу узнать про курс по Python",
#         question="Хочу узнать про курс по Python",
#         summary_history="",
#         summary_exact=""
#     )
#     print(result)



# --- Основная функция
def get_seller_answer(histories: list, debug_mode=False):
    """
    Получение ответа менеджера на основе истории переписки.
    """
    if not histories:
        return ""

    inputs = {
        "histories": histories,
        "last message from manager": histories[-2] if len(histories) >= 2 else "",
        "last message from client": histories[-1],
        "last client-manager qa": "\n".join(histories[-3:-1]),
        # "search_index": search_index,
    }

    greeting_identification_chain = create_greeting_chain()
    parallel_extractors = build_parallel_extractors(db_name="", debug_mode=debug_mode)
    summary_chain = RunnableLambda(create_summary_exact)
    router_chain = create_router_chain(debug_mode=debug_mode)
    experts_chain = RunnableLambda(process_experts)


    full_chain = RunnableSequence(
        greeting_identification_chain,
        parallel_extractors,
        summary_chain,
        router_chain,
        experts_chain
    )

    results = full_chain.invoke(inputs)

    if debug_mode:
        print("Результаты экстракции:", results)
        for name, result in  results.items():
            if isinstance(result, AIMessage):
                print(name, " - ", result.content)  # Выводим content
                # print(
                #     f"Token usage for {name}: {result.response_metadata.get('token_usage', 'N/A')}")  # Выводим token_usage
                # print(f"Usage metadata for {name}: {result.usage_metadata}")  # Выводим usage_metadata


    return results


# === Пример запуска ===
if __name__ == "__main__":
    histories = ["Клиент: Хочу узнать про курс Python",
                 "Менеджер: Это отличный курс по Python.Добрый день! У нас есть два варианта: Базовый тариф — 15 000 ₽. Продвинутый тариф — 25 000 ₽. В оба тарифа входит полный доступ к материалам и поддержка.",
                 "Клиент: Мне кажется, это дорого, и я не уверен, что у меня будет время на обучение.",
                 "Понимаю ваши опасения. Давайте уточню — обучение можно проходить в удобное для вас время, в записи. Это поможет вам совмещать курс с основной работой.",
                 "Ну, возможно… Но я слышал, что многие онлайн-курсы оказываются бесполезными, и люди жалеют о потраченных деньгах."]
    neuro_data = get_seller_answer(histories, debug_mode=True)

