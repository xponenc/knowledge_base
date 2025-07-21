# retrieval_engine.py
import os
import re

from django.contrib.postgres.search import TrigramSimilarity
from django.db.models import Q, Value, TextField
from django.db.models.functions import Cast
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain, create_stuff_documents_chain
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.router import MultiRetrievalQAChain
from threading import Lock

from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain
from langchain.chains.router.multi_retrieval_prompt import MULTI_RETRIEVAL_ROUTER_TEMPLATE
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import BaseRetriever, Document
from langchain.chains import MultiRetrievalQAChain
import re
from typing import List, Tuple, Optional, Any, Dict

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from openai import OpenAI
import langchain

from app_chunks.models import Chunk, ChunkStatus

langchain.debug = False

from app_core.models import KnowledgeBase
from app_embeddings.services.embedding_config import FAISS_THRESHOLD, TOP_N
from app_embeddings.services.embedding_store import RERANKER, get_vectorstore, load_embedding
from knowledge_base.settings import BASE_DIR

_lock = Lock()
_multi_chain_cache = {}
_ensemble_chain_cache = {}


def get_cached_multi_chain(kb_id):
    key = f"kb_{kb_id}"
    with _lock:
        if key not in _multi_chain_cache:
            _multi_chain_cache[key] = init_multi_retrieval_qa_chain(kb_id)
        return _multi_chain_cache[key]


def get_cached_ensemble_chain(kb_id):
    key = f"kb_{kb_id}"
    with _lock:
        if key not in _ensemble_chain_cache:
            _ensemble_chain_cache[key] = init_ensemble_retriever_chain(kb_id)
        return _ensemble_chain_cache[key]


# Кастомный ретривер
class CustomRetriever(BaseRetriever):
    db_index: Any
    system_prompt: str
    description: str

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:

        def extract_links(metadata: dict) -> List[str]:
            links = []
            for key, value in metadata.items():
                if key.startswith("files__documents") or key.startswith("files__images") or key.startswith(
                        "external_links"):
                    if isinstance(value, str):
                        links.append(value)
            return links

        # print(f"[*** CustomRetriever] {query=}")
        if isinstance(query, dict):
            query = query.get("input", "")
        docs_with_scores = self.db_index.similarity_search_with_score(query, k=10)
        # print(f"[*** CustomRetriever] {docs_with_scores=}")

        top_docs = rerank_documents(query, docs_with_scores, threshold=1.5)
        # print(f"[*** CustomRetriever] {top_docs=}")

        if not top_docs:
            return [Document(page_content="Пожалуйста, задайте вопрос иначе или уточните его.")]
        for doc, score in top_docs:
            doc.metadata["retriever_score"] = float(score)
        top_docs = [doc for doc, _ in top_docs]
        # Добавляем полезные ссылки из metadata
        enriched_docs = self._append_links_to_documents(top_docs)

        return enriched_docs


    def _append_links_to_documents(self, docs: List[Document]) -> List[Document]:
        def extract_links(metadata: dict) -> List[str]:
            links = []
            for key, value in metadata.items():
                if key.startswith("files__documents") or key.startswith("files__images") or key.startswith(
                        "external_links"):
                    if isinstance(value, str):
                        links.append(value)
            return links

        for doc in docs:
            links = extract_links(doc.metadata)
            if links:
                links_text = "\n\nПолезные материалы:\n" + "\n".join(f"- {link}" for link in links)
                doc.page_content += links_text  # Модифицируем содержимое документа

        return docs

# Кастомная цепочка RetrievalQA с постобработкой
class CustomRetrievalQA(RetrievalQA):
    def _get_docs(self, inputs, run_manager: CallbackManagerForChainRun = None) -> List[Document]:
        # print(f"[*** CustomRetrievalQA] {inputs=}")

        query = inputs.get("input") or inputs.get("query")
        docs = self.retriever.get_relevant_documents(query)
        # print(f"[*** CustomRetrievalQA] {docs=}")
        return docs


    def _call(self, inputs: dict, run_manager: CallbackManagerForChainRun = None) -> Dict[str, Any]:
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


def init_multi_retrieval_qa_chain(kb_id):
    kb = (KnowledgeBase.objects
          .select_related("engine")
          .prefetch_related("website_set", "cloudstorage_set", "localstorage_set", "urlbatch_set")
          .get(pk=kb_id))
    embedding_engine = kb.engine
    embeddings_model_name = embedding_engine.model_name

    try:
        embeddings_model = load_embedding(embeddings_model_name)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки модели {embeddings_model_name}: {str(e)}")

    # Список всех наборов хранилищ
    storage_sets = [
        kb.website_set,
        kb.cloudstorage_set,
        kb.localstorage_set,
        kb.urlbatch_set,
    ]

    destination_chains = {}
    default_chain = None
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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
                print(f"Ошибка загрузки векторного хранилища для {storage.__class__.__name__} id {storage.pk}: {str(e)}")
                continue

            retriever = CustomRetriever(
                db_index=db_index,
                system_prompt=kb.system_instruction,
                description=storage.description or storage.name
            )

            # prompt = PromptTemplate(
            #     input_variables=["system_prompt", "context", "input"],
            #     template="SYSTEM: {system_prompt}\n\nCONTEXT: {context}\n\nQuestion: {input}"
            # )
            # Версия для ChatOpenAI
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("{system_prompt}"),
                HumanMessagePromptTemplate.from_template("CONTEXT: {context}\n\nQuestion: {input}")
            ])
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

            chain = CustomRetrievalQA(
                retriever=retriever,
                combine_documents_chain=combine_documents_chain,
                return_source_documents=True
            )

            name = f"retriever_{storage.__class__.__name__}_{storage.pk}"
            destination_chains[name] = chain

            if storage.default_retriever:
                default_chain = chain

    if not destination_chains:
        raise ValueError("Не удалось создать цепочки поиска для базы знаний.")

    destinations_str = "\n".join([
        f"{name}: {destination_chains[name].retriever.description}"
        for name in destination_chains
    ])

    router_template = MULTI_RETRIEVAL_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(next_inputs_inner_key="query"),
    )

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    if default_chain is None:
        default_chain = ConversationChain(
            llm=llm,
            prompt=PromptTemplate(template="Answer: {input}", input_variables=["input"]),
            input_key="input",
            output_key="result"
        )

    multi_chain = MultiRetrievalQAChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True
    )
    return multi_chain


def init_ensemble_retriever_chain(kb_id: int, *, k: int = 5):
    kb = (KnowledgeBase.objects
          .select_related("engine")
          .prefetch_related("website_set", "cloudstorage_set", "localstorage_set", "urlbatch_set")
          .get(pk=kb_id))
    embedding_engine = kb.engine
    embeddings_model_name = embedding_engine.model_name

    try:
        embeddings_model = load_embedding(embeddings_model_name)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки модели {embeddings_model_name}: {str(e)}")

    # Список всех наборов хранилищ
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
                print(
                    f"Ошибка загрузки векторного хранилища для {storage.__class__.__name__} id {storage.pk}: {str(e)}")
                continue
            # retriever = db_index.as_retriever(search_kwargs={"k": 2})
            retriever = create_custom_retriever(db_index, k=2)
            retrievers.append(retriever)

    if not retrievers:
        raise ValueError("Не удалось создать EnsembleRetriever, нет доступных векторных хранилищ")

    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers, weights=[1.0] * len(retrievers),
    )

    # Оборачиваем EnsembleRetriever в RunnableLambda для реранкинга и добавления score
    def rerank_and_enrich_documents(input):
        query = input.get("input", input) if isinstance(input, dict) else input
        # Получаем все документы от EnsembleRetriever
        docs = ensemble_retriever.invoke(query)
        # print(f"rerank_and_enrich_documents {docs=}")

        # Собираем документы с их исходными score для реранкинга
        docs_and_scores = [(doc, doc.metadata.get("retriever_score", 0.0)) for doc in docs]

        # Применяем реранкинг
        top_docs = rerank_documents(query, docs_and_scores, threshold=1.5)

        if not top_docs:
            return [Document(page_content="Пожалуйста, задайте вопрос иначе или уточните его.")]

        # Добавляем retriever_score и ссылки в метаданные
        updated_docs = []
        for doc, score in top_docs:
            new_metadata = doc.metadata.copy()
            new_metadata["retriever_score"] = float(score)
            updated_doc = Document(
                page_content=doc.page_content,
                metadata=new_metadata
            )
            updated_docs.append(updated_doc)

        # Добавляем ссылки из метаданных
        updated_docs = append_links_to_documents(updated_docs)

        return updated_docs

    scored_retriever = RunnableLambda(rerank_and_enrich_documents)

    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=CustomCrossEncoderReranker(cross_encoder_model=RERANKER),
    #     base_retriever=ensemble_retriever
    # )

    system_prompt = getattr(kb, "system_instruction")
    if not system_prompt:
        raise ValueError("Не удалось создать EnsembleRetriever, нет system_prompt")

    # prompt = ChatPromptTemplate.from_messages([
    #     SystemMessagePromptTemplate.from_template("{system_prompt}"),
    #     # SystemMessagePromptTemplate.from_template(system_prompt),
    #     HumanMessagePromptTemplate.from_template("CONTEXT: {context}\n\nQuestion: {input}")
    # ])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human", "CONTEXT: {context}\n\nQuestion: {input}")
    ])

    # llm = ChatOpenAI(model=model, temperature=temperature)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    # combine_documents_chain = StuffDocumentsChain(
    #     llm_chain=llm_chain,
    #     document_variable_name="context"
    # )
    document_chain = create_stuff_documents_chain(llm, prompt)

    # qa_chain = RetrievalQA(
    #     retriever=ensemble_retriever,
    #     combine_documents_chain=combine_documents_chain,
    #     return_source_documents=True,
    #     input_key="input",
    #     output_key="result"
    # )
    retrieval_chain = create_retrieval_chain(scored_retriever, document_chain)

    return retrieval_chain


def create_custom_retriever(vectorstore: FAISS, k: int = 2):
    """Создаёт кастомный retriever, который добавляет score в метаданные документов."""
    class CustomRetriever(Runnable):
        def __init__(self, vectorstore, k):
            super().__init__()
            self.vectorstore = vectorstore
            self.k = k

        def invoke(self, input, config=None, **kwargs):
            # Выполняем поиск с возвратом score
            docs_and_scores = self.vectorstore.similarity_search_with_score(input, k=self.k)
            # Добавляем score в метаданные каждого документа
            updated_docs = []
            for doc, score in docs_and_scores:
                new_metadata = doc.metadata.copy()
                new_metadata["retriever_score"] = float(score)
                updated_doc = Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
                updated_docs.append(updated_doc)
            enriched_documents = append_links_to_documents(updated_docs)
            return enriched_documents

        def get_relevant_documents(self, query: str):
            return self.invoke(query)

    return CustomRetriever(vectorstore, k)


def rerank_documents(query, docs_with_scores, reranker=RERANKER, threshold=FAISS_THRESHOLD, top_n=TOP_N):
    """
     Фильтрует и переупорядочивает список документов на основе релевантности к запросу.

     Сначала отфильтровывает документы, у которых score (расстояние от запроса)
     ниже заданного порога. Затем применяет reranker-модель (CrossEncoder),
     которая переоценивает релевантность по содержимому документа и запроса.
     Возвращает топ-N документов с наивысшими оценками reranker-а.

     Args:
         query (str): Текстовый запрос пользователя.
         docs_with_scores (List[Tuple[Document, float]]): Список документов с исходными расстояниями (например, cosine distance).
         reranker (CrossEncoder): Модель повторного ранжирования, сравнивающая (запрос, документ).
         threshold (float): Пороговое значение расстояния, выше которого документы отбрасываются.
         top_n (int): Количество документов, которые нужно вернуть после повторного ранжирования.

     Returns:
         List[Tuple[Document, float]]: Список топ-N документов с их оценками (по умолчанию исходный FAISS score).
     """
    # Фильтрация по порогу
    # print(f"[*** rerank_documents] {query=}")
    # print(f"[*** rerank_documents] {docs_with_scores=}")

    filtered = [(doc, score) for doc, score in docs_with_scores if score < threshold]
    if not filtered:
        filtered = [(doc, score) for doc, score in docs_with_scores if score < threshold + 0.4]
        if not filtered:
            return []
    # Подготовка пар (вопрос, документ) и получение rerank-оценок
    pairs = [(query, doc.page_content) for doc, _ in filtered]
    scores = reranker.predict(pairs)

    # Сортировка по убыванию релевантности
    reranked = sorted(zip(filtered, scores), key=lambda x: x[1], reverse=True)

    # Возвращаем [(Document, score)] (используем старый score из FAISS, если нужно можно заменить rerank-оценкой)
    return [doc_with_score for doc_with_score, _ in reranked[:top_n]]


def reformulate_question(
    current_question: str,
    chat_history: List[Tuple[str, str]],
    openai_model: str = "gpt-4o-mini",
) -> str:
    """
    Переформулирует текущий вопрос с учётом истории диалога,
    только если он логически связан с предыдущими репликами.

    Если вопрос самостоятельный и тема изменилась, возвращает оригинальный вопрос.

    Args:
        current_question (str): Новый вопрос пользователя.
        chat_history (List[Tuple[str, str]]): История диалога как список пар (вопрос, ответ).
        openai_model (str): Название модели OpenAI (по умолчанию "gpt-4o-mini").

    Returns:
        str: Переформулированный вопрос (если связан с историей) или оригинальный вопрос.

    Пример:
        chat_history = [
            ("Кто у вас генеральный директор?", "Иванов Иван Иванович"),
        ]
        current_question = "А есть приказ о его назначении?"
        reformulated = reformulate_question(current_question, chat_history)
    """
    # Собираем историю диалога
    chat_str = ""
    for user_q, ai_a in chat_history:
        chat_str += f"Пользователь: {user_q}\nАссистент: {ai_a}\n"

    # Шаблон prompt-а
    reformulate_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
            История диалога:
            {chat_history}
            
            Вопрос пользователя: {question}
            
            1. Самодостаточен ли этот вопрос для поиска в RAG исходя из контекста истории? Ответь "да" или "нет".
            2. Если "нет", переформулируй вопрос так, чтобы он стал самостоятельным и включал важный контекст из истории.
            3. Если "да", верни исходный вопрос без изменений.
            
            Ответ: Только новый или исходный вопрос, без комментариев
        """,
    )
    # 1. Связан ли этот вопрос с историей диалога? Ответь "да" или "нет".
    # Инициализируем модель
    llm = ChatOpenAI(temperature=0, model=openai_model)
    llm_chain = LLMChain(llm=llm, prompt=reformulate_prompt)

    # Выполняем запрос
    response = llm_chain.run({
        "chat_history": chat_str.strip(),
        "question": current_question.strip()
    }).strip()

    print(response)
    return response
    # Попытка найти переформулированный текст после "2." или просто вернуть оригинальный
    # lines = response.split("\n")
    # reformulated = None
    # for line in lines:
    #     if line.strip().startswith("2."):
    #         reformulated = line.strip()[2:].strip(":").strip()
    #         break
    # return reformulated if reformulated else current_question


# def answer_index(system, query, verbose=False):
def answer_index(db_index, system, query, verbose=False):
    docs_with_scores = db_index.similarity_search_with_score(query, k=10)
    if verbose:
        print("Вывод docs_with_scores ========= ")
        for doc in docs_with_scores:
            print("\n", doc)
        print("========= \n\n\n")
    top_docs = docs_with_scores
    # Реранкинг
    top_docs = rerank_documents(query, docs_with_scores)
    if verbose:
        print("Вывод top_docs ========= ")
        for doc in top_docs:
            print("\n", doc)
        print("========= \n\n\n")

    if not top_docs:
        return top_docs, "Пожалуйста, задайте вопрос иначе или уточните его."

    # message_content = re.sub(
    #     r'\n{2}', ' ',
    #     '\n '.join([f'\nОтрывок документа №{i + 1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(top_docs)])
    # )
    message_content = re.sub(
        r'\n{2}', ' ',
        '\n '.join([
            f'\nОтрывок документа №{i + 1}\n=====================' + doc.page_content + '\n'
            for i, (doc, _) in enumerate(top_docs)
        ])
    )

    if verbose:
        print("message_content:\n", message_content)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Ответь на вопрос. Документ с информацией для ответа: {message_content}\n\nВопрос пользователя: \n{query}"}
    ]

    client = OpenAI()
    completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
    return top_docs, completion.choices[0].message.content


def answer_index_with_metadata(db_index, system, query, verbose=False, metadata=True):
    docs_with_scores = db_index.similarity_search_with_score(query, k=5)
    if verbose:
        print("Вывод docs_with_scores ========= ")
        for doc in docs_with_scores:
            print("\n", doc)
        print("========= \n\n\n")

    # # Реранкинг
    top_docs = rerank_documents(query, docs_with_scores)
    if verbose:
        print("Вывод top_docs ========= ")
        for doc in top_docs:
            print("\n", doc)
        print("========= \n\n\n")

    if not top_docs:
        return top_docs, "Пожалуйста, задайте вопрос иначе или уточните его."

    message_blocks = []
    for i, (doc, score) in enumerate(top_docs):
        meta = doc.metadata or {}
        metadata_block = ""

        if metadata:
            # Формируем часть с метаданными
            internal_links = "\n".join(
                f"- {v}" for k, v in meta.items() if k.startswith("internal_links__")
            )
            external_links = "\n".join(
                f"- {v}" for k, v in meta.items() if k.startswith("external_links__")
            )
            images = "\n".join(
                f"- {v}" for k, v in meta.items() if k.startswith("files__images__")
            )
            documents = "\n".join(
                f"- {v}" for k, v in meta.items() if k.startswith("files__documents__")
            )


            if internal_links:
                metadata_block += f"\n[Внутренние ссылки по теме]:\n{internal_links}"
            if external_links:
                metadata_block += f"\n[Ссылки на внешние ресурсы]:\n{external_links}"
            if images:
                metadata_block += f"\n[Изображения]:\n{images}"
            if documents:
                metadata_block += f"\n[Документы]:\n{documents}"

        message_blocks.append(
            f"""
                Отрывок документа №{i + 1}
                =====================
                {doc.page_content}
                {metadata_block}
            """.strip()
        )

    message_content = "\n\n".join(message_blocks)

    if verbose:
        print("message_content:\n", message_content)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Ответь на вопрос. Документ с информацией для ответа: {message_content}\n\nВопрос пользователя: \n{query}"}
    ]

    client = OpenAI()
    completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
    return top_docs, completion.choices[0].message.content


def trigram_similarity_answer_index(kb_id, system, query, verbose=False):
    chunks = Chunk.objects.filter(
        Q(url_content__url__batch__kb_id=kb_id) |
        Q(url_content__url__site__kb_id=kb_id) |
        Q(raw_content__network_document__storage__kb_id=kb_id) |
        Q(raw_content__local_document__storage__kb_id=kb_id) |
        Q(cleaned_content__raw_content__network_document__storage__kb_id=kb_id) |
        Q(cleaned_content__raw_content__local_document__storage__kb_id=kb_id)
    ).filter(status=ChunkStatus.ACTIVE.value).distinct()

    chunks = chunks.annotate(
        similarity=TrigramSimilarity('page_content',  Cast(Value(query), TextField()))
    ).order_by('-similarity')[:10]

    docs_with_scores = []
    for chunk in chunks:
        chunk.metadata["score"] = chunk.similarity
        docs_with_scores.append((Document(page_content=chunk.page_content, metadata=chunk.metadata), chunk.similarity))

    if verbose:
        print("Вывод docs_with_scores ========= ")
        for doc in docs_with_scores:
            print("\n", doc)
        print("========= \n\n\n")
    # Реранкинг
    top_docs = rerank_documents(query, docs_with_scores)
    if verbose:
        print("Вывод top_docs ========= ")
        for doc in top_docs:
            print("\n", doc)
        print("========= \n\n\n")

    if not top_docs:
        return top_docs, "Пожалуйста, задайте вопрос иначе или уточните его."

    # message_content = re.sub(
    #     r'\n{2}', ' ',
    #     '\n '.join([f'\nОтрывок документа №{i + 1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(top_docs)])
    # )
    top_docs = [doc for (doc, _score) in top_docs]
    enriched_documents = append_links_to_documents(top_docs)

    message_content = re.sub(
        r'\n{2}', ' ',
        '\n '.join([
            f'\nОтрывок документа №{i + 1}\n=====================' + doc.page_content + '\n'
            for i, doc in enumerate(enriched_documents)
        ])
    )

    if verbose:
        print("message_content:\n", message_content)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Ответь на вопрос. Документ с информацией для ответа: {message_content}\n\nВопрос пользователя: \n{query}"}
    ]

    client = OpenAI()
    completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
    return top_docs, completion.choices[0].message.content


def append_links_to_documents(docs: List[Document]) -> List[Document]:
    def extract_links(metadata: dict) -> List[str]:
        links = []
        for key, value in metadata.items():
            if key.startswith("files__documents") or key.startswith("files__images") or key.startswith(
                    "external_links"):
                if isinstance(value, str):
                    links.append(value)
        return links

    for doc in docs:
        links = extract_links(doc.metadata)
        if links:
            links_text = "\n\nПолезные материалы:\n" + "\n".join(f"- {link}" for link in links)
            doc.page_content += links_text  # Модифицируем содержимое документа

    return docs