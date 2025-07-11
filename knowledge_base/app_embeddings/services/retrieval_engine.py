# retrieval_engine.py
import os
import re

from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.router import MultiRetrievalQAChain
from threading import Lock

from langchain.schema import BaseRetriever, Document
from langchain.chains import MultiRetrievalQAChain
import re
from typing import List, Tuple, Optional, Any

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

from app_core.models import KnowledgeBase
from app_embeddings.services.embedding_config import FAISS_THRESHOLD, TOP_N
from app_embeddings.services.embedding_store import RERANKER, get_vectorstore, load_embedding
from knowledge_base.settings import BASE_DIR

_lock = Lock()
_multi_chain_cache = {}


def get_cached_multi_chain(kb_id):
    key = f"kb_{kb_id}"
    with _lock:
        if key not in _multi_chain_cache:
            _multi_chain_cache[key] = init_multi_retrieval_qa_chain(kb_id)
        return _multi_chain_cache[key]


def init_multi_retrieval_qa_chain(kb_id):
    retriever_infos = []
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

    default_retriever = None

    # Обход всех хранилищ
    for storage_set in storage_sets:
        for storage in storage_set.all():
            # Формируем путь к FAISS индексу
            faiss_dir = os.path.join(
                BASE_DIR, "media", "kb", str(kb.pk), "embedding_stores",
                f"{storage.__class__.__name__}_id_{storage.pk}_embedding_store",
                f"{embedding_engine.name}_faiss_index_db"
            )
            try:
                db_index = get_vectorstore(
                    path=faiss_dir,
                    embeddings=embeddings_model
                )
            except Exception as e:
                print(f"Ошибка загрузки векторного хранилища для {storage.__class__.__name__} id {storage.pk}: {str(e)}")
                db_index = None

            if db_index:
                retriever_infos.append({
                    "name": storage.name,
                    "retriever": CustomRetriever(
                        db_index=db_index,
                        system_prompt=kb.system_instruction
                    ),
                    "description": storage.description,
                })
                if storage.default_retriever:
                    default_retriever = CustomRetriever(
                        db_index=db_index,
                        system_prompt=kb.system_instruction
                    )

    # retriever = db_simblie_merged.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 4}
    # )
    #
    # template = """Отвечайте на вопросы только на основе следующего контекста:
    #
    # {context}
    #
    # Вопрос: {question}
    # """
    # prompt = ChatPromptTemplate.from_template(template)
    # model = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)
    #
    # def format_docs(docs):
    #     return "\n\n".join([d.page_content for d in docs])
    #
    # chain = (
    #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #         | prompt
    #         | model
    #         | StrOutputParser()
    # )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_instruction = kb.system_instruction
    llm_template = f"SYSTEM: {system_instruction}\n\nCONTEXT: {{context}}\n\n Question: {{input}}"

    prompt = PromptTemplate(input_variables=["input", "context"], template=llm_template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    # Цепочка, которую будем использовать по умолчанию
    default_chain_llm = prompt | llm
    # Создаем MultiRetrievalQAChain
    multi_chain = MultiRetrievalQAChain.from_retrievers(
        llm = llm,
        retriever_infos=retriever_infos,
        # default_chain_llm=default_chain_llm,
        combine_documents_chain=combine_documents_chain,
        default_retriever=default_retriever,
        verbose=True,
    )
    return multi_chain


class SystemPromptChain(LLMChain):
    def __init__(self, llm, prompt, system_prompt):
        super().__init__(llm=llm, prompt=prompt)
        self.system_prompt = system_prompt

    def _call(self, inputs, **kwargs):
        inputs["system_prompt"] = self.system_prompt
        return super()._call(inputs, **kwargs)

class CustomRetriever(BaseRetriever):
    db_index: Any  # Используем Any, так как тип db_index зависит от вашей реализации (например, FAISS или Chroma)
    system_prompt: str

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs_with_scores = self.db_index.similarity_search_with_score(query, k=10)
        # if verbose:
        #     print("Вывод docs_with_scores ========= ")
        #     for doc in docs_with_scores:
        #         print("\n", doc)
        #     print("========= \n\n\n")
        # Реранкинг
        top_docs = rerank_documents(query, docs_with_scores)
        # if verbose:
        print("Вывод top_docs ========= ")
        for doc in top_docs:
            print("\n", doc)
        print("========= \n\n\n")

        if not top_docs:
            return [Document(page_content="Пожалуйста, задайте вопрос иначе или уточните его.")]

        # Формируем message_content
        message_content = re.sub(
            r'\n{2}', ' ',
            '\n '.join([
                f'\nОтрывок документа №{i + 1}\n=====================\n{doc.page_content}\n'
                for i, (doc, _) in enumerate(top_docs)
            ])
        )

        # Возвращаем ОДИН документ с полным контекстом
        return [Document(page_content=message_content)]


def rerank_documents(query, docs_with_scores, reranker=RERANKER, threshold=FAISS_THRESHOLD, top_n=TOP_N):
# def rerank_documents(query, docs_with_scores, reranker, threshold=FAISS_THRESHOLD, top_n=TOP_N):
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