import os
from typing import Dict, Optional, Any, List
from threading import Lock

import langchain
from django.conf import settings

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import StuffDocumentsChain, LLMChain, ConversationChain
from langchain.chains.router import MultiRetrievalQAChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.router.multi_retrieval_prompt import MULTI_RETRIEVAL_ROUTER_TEMPLATE
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain

from app_core.models import KnowledgeBase
from app_embeddings.services.embedding_store import load_embedding, get_vectorstore
from app_embeddings.services.retrieval_engine import rerank_documents

BASE_DIR = settings.BASE_DIR

langchain.debug = True

_lock = Lock()
_retriever_cache: Dict[str, Dict[str, Any]] = {}


class CustomRetriever(BaseRetriever):
    """
    Кастомный ретривер, оборачивающий FAISS-индекс и добавляющий:
    - повторную ранжировку документов
    - извлечение и добавление ссылок из metadata
    - поддержку system_prompt и описания
    """

    db_index: Any  # FAISS index с методом similarity_search_with_score()
    # system_prompt: str  # Инструкция для system_prompt, пробрасываемая далее
    description: str  # Краткое описание ретривера (для UI или маршрутизации)

    @staticmethod
    def extract_links(metadata: Dict[str, Any]) -> List[str]:
        """
        Извлекает ссылки из словаря metadata. Ожидает поля, начинающиеся на:
        - 'files__documents'
        - 'files__images'
        - 'external_links'

        :param metadata: словарь с метаинформацией документа
        :return: список строк-ссылок
        """
        links = []
        for key, value in metadata.items():
            if key.startswith(("files__documents", "files__images", "external_links")):
                if isinstance(value, str):
                    links.append(value)
        return links

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None
    ) -> List[Document]:
        """
        Выполняет поиск релевантных документов:
        - делает запрос к FAISS-индексу
        - повторно ранжирует документы с помощью кастомной функции
        - добавляет в документы ссылки из metadata

        :param query: текст запроса (или словарь, откуда будет извлечён ключ 'input')
        :param run_manager: (не используется, но требуется сигнатурой langchain)
        :return: список документов
        """
        if isinstance(query, dict):
            query = query.get("input", "")

        docs_with_scores = self.db_index.similarity_search_with_score(query, k=10)
        top_docs = rerank_documents(query, docs_with_scores, threshold=1.5)

        if not top_docs:
            return [Document(page_content="Пожалуйста, задайте вопрос иначе или уточните его.")]

        # Добавляем score в metadata
        for doc, score in top_docs:
            doc.metadata["retriever_score"] = float(score)

        docs = [doc for doc, _ in top_docs]
        enriched_docs = self._append_links_to_documents(docs)

        return enriched_docs

    def _append_links_to_documents(self, docs: List[Document]) -> List[Document]:
        """
        Добавляет ссылки из metadata к тексту каждого документа.

        :param docs: список документов
        :return: список документов с дополненными page_content
        """
        for doc in docs:
            links = self.extract_links(doc.metadata)
            if links:
                links_text = "\n\nПолезные материалы:\n" + "\n".join(f"- {link}" for link in links)
                doc.page_content += links_text
        return docs



class CustomRetrievalQA(RetrievalQA):
    def _get_docs(
        self,
        inputs: Dict[str, Any],
        run_manager: CallbackManagerForChainRun = None
    ) -> List[Document]:
        """
        Получает релевантные документы с помощью ретривера.
        Поддерживает ключи 'input' и 'query' для совместимости.
        """
        query = inputs.get("input") or inputs.get("query")
        return self.retriever.get_relevant_documents(query)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: CallbackManagerForChainRun = None
    ) -> Dict[str, Any]:
        """
        Основной метод выполнения цепочки:
        - Извлекает запрос и system_prompt
        - Получает документы
        - Передаёт их в цепочку объединения
        - Возвращает результат и исходные документы
        """
        query = inputs.get("input") or inputs.get("query")
        system_prompt = inputs.get("system_prompt", getattr(self.retriever, "system_prompt", ""))

        docs = self._get_docs(inputs, run_manager)

        new_inputs = {
            "input_documents": docs,
            "input": query,
            "system_prompt": system_prompt,
        }

        result = self.combine_documents_chain.invoke(new_inputs, run_manager=run_manager)

        return {
            "result": result["output_text"],
            "source_documents": docs
        }


class CustomMultiRetrievalQAChain(MultiRetrievalQAChain):
    """
    Расширение MultiRetrievalQAChain с поддержкой дополнительного поля `system_prompt`.

    Эта цепочка:
    - Использует `router_chain` для определения подходящей подцепочки на основе входного запроса.
    - Добавляет `system_prompt` в следующую цепочку, чтобы контролировать стиль ответа (например, шутливый).
    """

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        """
        Основной метод выполнения цепочки.

        Args:
            inputs (Dict[str, Any]): Входные данные, ожидается ключ `input` и опционально `system_prompt`.
            run_manager: Объект отслеживания выполнения цепочек (необязательный).

        Returns:
            Dict[str, Any]: Результат выполнения выбранной подцепочки.
        """
        # Извлекаем system_prompt из входных данных, если он есть
        system_prompt = inputs.get("system_prompt", "")

        # Запускаем маршрутизатор, чтобы определить, какую цепочку использовать
        router_output = self.router_chain.invoke(inputs, run_manager=run_manager)

        # Получаем имя целевой цепочки (если не задана — используется "DEFAULT")
        destination_name = router_output.get("destination", "DEFAULT")

        # Получаем изменённые входные данные для следующей цепочки
        next_inputs = router_output.get("next_inputs", {})

        # Добавляем system_prompt к следующему запросу
        next_inputs["system_prompt"] = system_prompt

        # Выбираем подходящую цепочку: либо найденную, либо дефолтную
        destination_chain = self.destination_chains.get(destination_name, self.default_chain)

        # Выполняем выбранную цепочку с модифицированными входными данными
        return destination_chain.invoke(next_inputs, run_manager=run_manager)


def get_cached_retrievers(kb_id: int) -> Dict[str, Any]:
    """
    Возвращает кэшированные ретриверы для базы знаний.
    Загружаются и кэшируются один раз для каждого kb_id.
    """
    key = f"kb_retrievers_{kb_id}"
    with _lock:
        if key not in _retriever_cache:
            _retriever_cache[key] = init_cached_retrievers(kb_id)
        return _retriever_cache[key]

def init_cached_retrievers(kb_id: int) -> Dict[str, Any]:
    """
    Инициализирует все доступные ретриверы для заданной базы знаний.
    Кешируется, потому что загрузка FAISS и Embedding дорогие.
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

    retrievers: Dict[str, CustomRetriever] = {}
    default_retriever: Optional[CustomRetriever] = None

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
                print(f"[WARN] Не удалось загрузить FAISS для {storage}: {e}")
                continue

            retriever = CustomRetriever(
                db_index=db_index,
                # system_prompt=kb.system_instruction,
                description=storage.description or storage.name
            )

            name = f"retriever_{storage.__class__.__name__}_{storage.pk}"
            retrievers[name] = retriever

            if storage.default_retriever:
                default_retriever = retriever

    if not retrievers:
        raise ValueError("Не удалось создать ни одного ретривера для базы знаний.")

    return {
        "retrievers": retrievers,
        "default": default_retriever,
        # "system_prompt": kb.system_instruction,
    }


def build_multi_chain(kb_id: int, llm: ChatOpenAI) -> CustomMultiRetrievalQAChain:
    """
    Собирает CustomMultiRetrievalQAChain с заданным LLM,
    используя ранее закешированные ретриверы.
    """
    cache = get_cached_retrievers(kb_id)
    retrievers: Dict[str, CustomRetriever] = cache["retrievers"]
    default_retriever: Optional[CustomRetriever] = cache["default"]
    # system_prompt: str = cache["system_prompt"]

    destination_chains = {}

    for name, retriever in retrievers.items():
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{system_prompt}"),
            HumanMessagePromptTemplate.from_template("CONTEXT: {context}\n\nQuestion: {input}")
        ])
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        combine_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

        destination_chains[name] = CustomRetrievalQA(
            retriever=retriever,
            combine_documents_chain=combine_chain,
            return_source_documents=True
        )

    if default_retriever:
        # Создаем цепочку для fallback
        default_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{system_prompt}"),
            HumanMessagePromptTemplate.from_template("CONTEXT: {context}\n\nQuestion: {input}")
        ])
        default_chain = CustomRetrievalQA(
            retriever=default_retriever,
            combine_documents_chain=StuffDocumentsChain(
                llm_chain=LLMChain(llm=llm, prompt=default_prompt),
                document_variable_name="context"
            ),
            return_source_documents=True
        )
    else:
        default_chain = ConversationChain(
            llm=llm,
            prompt=PromptTemplate(template="Answer: {input}", input_variables=["input"]),
            input_key="input",
            output_key="result"
        )

    destinations_str = "\n".join([
        f"{name}: {retrievers[name].description}" for name in destination_chains
    ])
    router_prompt = PromptTemplate(
        template=MULTI_RETRIEVAL_ROUTER_TEMPLATE.format(destinations=destinations_str),
        input_variables=["input"],
        output_parser=RouterOutputParser(next_inputs_inner_key="query"),
    )

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    return CustomMultiRetrievalQAChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True
    )