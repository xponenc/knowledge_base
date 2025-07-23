from typing import Dict, List, Any
from threading import Lock

from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate

from core.embedding import load_embedding, get_vectorstore
from core.rerank import rerank_documents, append_links_to_documents
from models.kb import KnowledgeBase
from retrievers.ensemble import EnsembleRetriever
from retrievers.utils import create_custom_retriever

from langchain.chat_models import ChatOpenAI

import os

_ensemble_chain_cache: Dict[str, Any] = {}
_lock = Lock()


def get_cached_ensemble_chain(kb_id: int, *, llm: ChatOpenAI | None = None) -> Any:
    """
    Получает (и кэширует) цепочку EnsembleRetriever для указанной базы знаний.
    Позволяет указать кастомную LLM на вызове, но не кэширует для каждой LLM отдельно.

    :param kb_id: ID базы знаний
    :param llm: (опционально) кастомная LLM
    :return: Цепочка LangChain, готовая к запуску
    """
    key = f"kb_{kb_id}"
    with _lock:
        if key not in _ensemble_chain_cache:
            retriever, system_prompt = build_ensemble_retriever(kb_id)
            _ensemble_chain_cache[key] = (retriever, system_prompt)

        retriever, system_prompt = _ensemble_chain_cache[key]
        llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "CONTEXT: {context}\n\nQuestion: {input}")
        ])
        document_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, document_chain)

        return chain


def build_ensemble_retriever(kb_id: int, *, k: int = 5) -> tuple[RunnableLambda, str]:
    """
    Строит EnsembleRetriever на базе всех хранилищ указанной базы знаний,
    объединяя их с весами, реранкингом и enrich-документами.

    :param kb_id: ID базы знаний
    :param k: количество ближайших документов для каждого ретривера
    :return: Tuple из финального ретривера и системной инструкции
    """
    kb = (
        KnowledgeBase.objects
        .select_related("engine")
        .prefetch_related(
            "website_set", "cloudstorage_set", "localstorage_set", "urlbatch_set"
        )
        .get(pk=kb_id)
    )

    embedding_engine = kb.engine
    embeddings_model_name = embedding_engine.model_name

    try:
        embeddings_model = load_embedding(embeddings_model_name)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки модели {embeddings_model_name}: {str(e)}")

    # Сбор всех хранилищ
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
                "media", "kb", str(kb.pk), "embedding_stores",
                f"{storage.__class__.__name__}_id_{storage.pk}_embedding_store",
                f"{embedding_engine.name}_faiss_index_db"
            )

            try:
                db_index = get_vectorstore(path=faiss_dir, embeddings=embeddings_model)
            except Exception as e:
                print(
                    f"Ошибка загрузки векторного хранилища для {storage.__class__.__name__} id {storage.pk}: {str(e)}")
                continue

            retriever = create_custom_retriever(db_index, k=2)
            retrievers.append(retriever)

    if not retrievers:
        raise ValueError("Не удалось создать EnsembleRetriever — нет доступных ретриверов.")

    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers,
        weights=[1.0] * len(retrievers)
    )

    def rerank_and_enrich_documents(input: str | Dict[str, Any]) -> List[Document]:
        """
        Оборачивает EnsembleRetriever:
        - получает документы
        - реранжирует
        - добавляет score и полезные ссылки
        """
        query = input.get("input", input) if isinstance(input, dict) else input
        docs = ensemble_retriever.invoke(query)
        docs_and_scores = [(doc, doc.metadata.get("retriever_score", 0.0)) for doc in docs]
        top_docs = rerank_documents(query, docs_and_scores, threshold=1.5)

        if not top_docs:
            return [Document(page_content="Пожалуйста, задайте вопрос иначе или уточните его.")]

        enriched_docs = []
        for doc, score in top_docs:
            doc.metadata["retriever_score"] = float(score)
            enriched_docs.append(doc)

        return append_links_to_documents(enriched_docs)

    final_retriever = RunnableLambda(rerank_and_enrich_documents)

    if not kb.system_instruction:
        raise ValueError("Не указана системная инструкция в базе знаний")

    return final_retriever, kb.system_instruction
