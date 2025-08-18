from functools import partial
from pprint import pprint

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough
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
from neuro_salesman.context import search_with_retriever
from neuro_salesman.expert import build_parallel_experts
from neuro_salesman.extractor import build_parallel_extractors
from neuro_salesman.greetings import create_remove_greeting_chain, create_extract_greeting_chain
from neuro_salesman.roles_config import EXTRACTOR_ROLES
from neuro_salesman.router import create_router_chain
from neuro_salesman.senior import create_senior_chain
from neuro_salesman.stylist import create_stylist_chain
from neuro_salesman.summary import create_summary_exact
from neuro_salesman.utils import debug_inputs, unpack_original_inputs, unpack_sequential
from utils.setup_logger import setup_logger

# --- Загрузка ключей
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

logger = setup_logger(name=__file__, log_dir="logs/neuro_salesman", log_file="ns.log")


# --- Основная функция
def get_seller_answer(
    session_type:str,
    session_id: str,
    kb_id: int,
    histories: list,
    debug_mode=False):
    """
    Получение ответа менеджера на основе истории переписки.
    """
    logger.info(f"[{session_type}:{session_id}] started")
    if not histories:
        logger.warning(f"[{session_type}:{session_id}] stopped. Empty histories parameter")
        return ""

    inputs = {
        "session_type": session_type,
        "session_id": session_id,
        "histories": histories,
        "last message from manager": histories[-2] if len(histories) >= 2 else "",
        "last message from client": histories[-1],
        "last client-manager qa": "\n".join(histories[-3:-1]),
        # "search_index": search_index,
    }
    logger.info(f"[{session_type}:{session_id}] Constructing greeting_identification_chain")
    # 1. Выявление приветствия
    greeting_identification_chain = create_extract_greeting_chain(debug_mode=debug_mode)

    logger.info(f"[{session_type}:{session_id}] Constructing parallel_extractors")
    # 2. Параллельная цепочка из экстракторов EXTRACTOR_ROLES
    parallel_extractors = build_parallel_extractors(debug_mode=debug_mode)

    logger.info(f"[{session_type}:{session_id}] Constructing summary_chain")
    # 3. Саммаризация истории чата
    summary_chain = RunnableLambda(partial(create_summary_exact, debug_mode=debug_mode))

    logger.info(f"[{session_type}:{session_id}] Constructing router_chain")
    # 4. Создание списка роутинга их агентов-экспертов EXPERTS_ROLES
    router_chain = create_router_chain(debug_mode=debug_mode)

    logger.info(f"[{session_type}:{session_id}] Constructing sequential_summary_and_router_chains")
    #  5. Последовательная цепочка для summary_chain и router_chain
    sequential_summary_and_router_chains = RunnableSequence(
        summary_chain,
        router_chain
    )

    logger.info(f"[{session_type}:{session_id}] Constructing parallel_chains")
    # 6. Параллельная цепочка из (Последовательная цепочка для summary_chain и router_chain) и запроса поиска
    # контента в базе знаний плюс проброс через цепочку входных inputs через RunnablePassthrough()
    parallel_chains = RunnableParallel(
        summary_and_router=sequential_summary_and_router_chains,
        search_index=RunnableLambda(search_with_retriever),
        original_inputs=RunnablePassthrough()
    )

    logger.info(f"[{session_type}:{session_id}] Constructing experts_chain")
    # 7. Параллельная цепочка из экспертов EXPERTS_ROLES
    experts_chain = build_parallel_experts(debug_mode=debug_mode)

    logger.info(f"[{session_type}:{session_id}] Constructing senior_chain")
    # 8. Формирование итогового ответа старшим агентом
    senior_chain = create_senior_chain(debug_mode=debug_mode)

    logger.info(f"[{session_type}:{session_id}] Constructing stylist_chain")
    # 9. Стилизация итогового ответа
    stylist_chain = create_stylist_chain(debug_mode=debug_mode)

    logger.info(f"[{session_type}:{session_id}] Constructing remove_greeting_chain")
    # 10. Контрольное удаление приветствия из итогового ответа
    remove_greeting_chain = create_remove_greeting_chain(debug_mode=debug_mode)

    full_chain = RunnableSequence(
        greeting_identification_chain,
        # RunnableLambda(lambda x: debug_inputs(x, "After Greeting")),
        parallel_extractors,
        RunnableLambda(unpack_original_inputs),
        # RunnableLambda(lambda x: debug_inputs(x, "After Parallel Extractors")),
        parallel_chains,
        RunnableLambda(unpack_original_inputs),
        RunnableLambda(unpack_sequential),
        # RunnableLambda(lambda x: debug_inputs(x, "After Inputs Extractors")),
        experts_chain,
        RunnableLambda(unpack_original_inputs),
        senior_chain,
        # RunnableLambda(lambda x: debug_inputs(x, "After Senior Chain")),
        stylist_chain,
        # RunnableLambda(lambda x: debug_inputs(x, "After Stylist Chain")),
        remove_greeting_chain,
        RunnableLambda(lambda x: debug_inputs(x, "After Remove Greeting Chain"))
    )

    results = full_chain.invoke(inputs)

    if debug_mode:
        print("Результаты экстракции:", results)
        for name, result in results.items():
            if isinstance(result, AIMessage):
                print(name, " - ", result.content)  # Выводим content
                # print(
                #     f"Token usage for {name}: {result.response_metadata.get('token_usage', 'N/A')}")  # Выводим token_usage
                # print(f"Usage metadata for {name}: {result.usage_metadata}")  # Выводим usage_metadata

    return results


# === Пример запуска ===
if __name__ == "__main__":
    session_type = "web"
    session_id = "123456"
    kb_pk = 1
    histories = ["Клиент: Хочу узнать про курс Python",
                 "Менеджер: Это отличный курс по Python.Добрый день! У нас есть два варианта: Базовый тариф — 15 000 ₽. Продвинутый тариф — 25 000 ₽. В оба тарифа входит полный доступ к материалам и поддержка.",
                 "Клиент: Мне кажется, это дорого, и я не уверен, что у меня будет время на обучение.",
                 "Менеджер: Понимаю ваши опасения. Давайте уточню — обучение можно проходить в удобное для вас время, в записи. Это поможет вам совмещать курс с основной работой.",
                 "Клиент: Ну, возможно… Но я слышал, что многие онлайн-курсы оказываются бесполезными, и люди жалеют о потраченных деньгах."]
    neuro_data = get_seller_answer(
        session_type,
        session_id,
        kb_pk,
        histories,
        debug_mode=True)
