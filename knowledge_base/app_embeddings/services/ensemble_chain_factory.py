import os
import sys
import logging
from threading import Lock
from typing import Dict, Any, Optional

from django.conf import settings
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.runnables import Runnable, RunnableLambda
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers.ensemble import EnsembleRetriever

from app_embeddings.services.embedding_store import get_vectorstore, load_embedding
from app_embeddings.services.retrieval_engine import rerank_documents, append_links_to_documents
from app_core.models import KnowledgeBase

logger = logging.getLogger(__name__)

BASE_DIR = settings.BASE_DIR
_lock = Lock()

# Кеш для retriever'ов, тяжелая часть (FAISS, Embeddings)
_ensemble_retriever_cache: Dict[str, Runnable] = {}


def build_ensemble_chain(kb_id: int, llm: ChatOpenAI) -> Runnable:
    """
    Собирает цепочку на лету: prompt + LLM + retrieval chain.
    Использует предварительно закешированный EnsembleRetriever.

    Args:
        kb_id (int): ID базы знаний.
        llm (ChatOpenAI): Инстанс LLM (например, GPT-4o).

    Returns:
        Runnable: Готовая к вызову цепочка, возвращающая стандартный словарь.
    """
    retriever = get_cached_ensemble_retriever(kb_id)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system_prompt}"),
        HumanMessagePromptTemplate.from_template("CONTEXT: {context}\n\nQuestion: {input}")
    ])

    # Создаем retrieval chain, которая возвращает словарь с ключами "input", "context", "answer"
    retrieval_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

    return retrieval_chain


def get_cached_ensemble_retriever(kb_id: int) -> Runnable:
    """
    Возвращает кешированный EnsembleRetriever с реранкингом и enrichment.

    Args:
        kb_id (int): ID базы знаний.

    Returns:
        Runnable: Обёртка вокруг retriever с enrichment.
    """
    key = f"ensemble_retriever_{kb_id}"
    with _lock:
        if key not in _ensemble_retriever_cache:
            _ensemble_retriever_cache[key] = init_cached_ensemble_retriever(kb_id)
        return _ensemble_retriever_cache[key]


def init_cached_ensemble_retriever(kb_id: int, *, k: int = 2) -> Runnable:
    """
    Инициализирует EnsembleRetriever, объединяя все доступные FAISS-индексы для базы знаний.
    Добавляет реранкинг и enrichment документов.

    Args:
        kb_id (int): ID базы знаний.
        k (int): Кол-во ближайших соседей в каждом индексе.

    Returns:
        Runnable: Обёртка вокруг EnsembleRetriever с enrichment.
    """
    kb = (
        KnowledgeBase.objects
        .select_related("engine")
        .prefetch_related("website_set", "cloudstorage_set", "localstorage_set", "urlbatch_set")
        .get(pk=kb_id)
    )
    embedding_engine = kb.engine
    embeddings_model_name = embedding_engine.model_name

    try:
        embeddings_model = load_embedding(embeddings_model_name)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки модели эмбеддинга {embeddings_model_name}: {str(e)}")

    storage_sets = [
        kb.website_set,
        kb.cloudstorage_set,
        kb.localstorage_set,
        kb.urlbatch_set,
    ]

    retrievers = []
    for storage_set in storage_sets:
        for storage in storage_set.all():
            faiss_dir = os.path.join(
                BASE_DIR, "media", "kb", str(kb.pk), "embedding_stores",
                f"{storage.__class__.__name__}_id_{storage.pk}_embedding_store",
                f"{embedding_engine.name}_faiss_index_db"
            )
            try:
                db_index = get_vectorstore(path=faiss_dir, embeddings=embeddings_model)
            except Exception as e:
                logger.warning(f"FAISS загрузка не удалась для {storage}: {e}")
                continue

            retriever = create_custom_retriever(db_index, k=k)
            retrievers.append(retriever)

    if not retrievers:
        raise ValueError("Не удалось создать EnsembleRetriever: нет доступных векторных индексов.")

    ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=[1.0] * len(retrievers))

    def rerank_and_enrich_documents(input):
        query = input.get("input", input) if isinstance(input, dict) else input
        docs = ensemble_retriever.invoke(query)
        docs_and_scores = [(doc, doc.metadata.get("retriever_score", 0.0)) for doc in docs]

        top_docs = docs_and_scores
        # top_docs = rerank_documents(query, docs_and_scores, threshold=1.5) # TODO Rerank выключен

        if not top_docs:
            return [Document(page_content="Пожалуйста, задайте вопрос иначе или уточните его.")]

        enriched = []
        for doc, score in top_docs:
            doc.metadata["retriever_score"] = float(score)
            enriched.append(doc)

        return append_links_to_documents(enriched)

    return RunnableLambda(rerank_and_enrich_documents)


def create_custom_retriever(vectorstore, k: int = 2):
    """
    Создаёт retriever с поддержкой извлечения score и enrichment ссылками.

    Args:
        vectorstore: FAISS-индекс
        k (int): Кол-во ближайших соседей

    Returns:
        Runnable: Обёртка вокруг vectorstore с обогащением.
    """

    class CustomRetriever(Runnable):
        def __init__(self, vectorstore, k):
            self.vectorstore = vectorstore
            self.k = k

        def invoke(self, input, config=None, **kwargs):
            docs_and_scores = self.vectorstore.similarity_search_with_score(input, k=self.k)
            updated_docs = []
            for doc, score in docs_and_scores:
                doc.metadata["retriever_score"] = float(score)
                updated_docs.append(doc)
            return append_links_to_documents(updated_docs)

        def get_relevant_documents(self, query: str):
            return self.invoke(query)

    return CustomRetriever(vectorstore, k)
