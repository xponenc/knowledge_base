# retrieval_engine.py
import os
import re

from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.router import MultiRetrievalQAChain
from threading import Lock

from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain
from langchain.chains.router.multi_retrieval_prompt import MULTI_RETRIEVAL_ROUTER_TEMPLATE
from langchain.schema import BaseRetriever, Document
from langchain.chains import MultiRetrievalQAChain
import re
from typing import List, Tuple, Optional, Any, Dict

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
import langchain
langchain.debug = True

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


# Кастомный ретривер
class CustomRetriever(BaseRetriever):
    db_index: Any
    system_prompt: str

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        if isinstance(query, dict):
            query = query.get("input", "")
        docs_with_scores = self.db_index.similarity_search_with_score(query, k=10)
        top_docs = rerank_documents(query, docs_with_scores)
        if not top_docs:
            return [Document(page_content="Пожалуйста, задайте вопрос иначе или уточните его.")]
        return [doc for doc, _ in top_docs]


# Кастомная цепочка RetrievalQA с постобработкой
class CustomRetrievalQA(RetrievalQA):
    def _get_docs(self, inputs, run_manager: CallbackManagerForChainRun = None) -> List[Document]:
        # docs = super()._get_docs(inputs, run_manager=run_manager)
        # if not docs or docs[0].page_content == "Пожалуйста, задайте вопрос иначе или уточните его.":
        #     return docs
        # processed_docs = []
        # for doc in docs:
        #     # Пример постобработки: добавление метаданных
        #     # doc.metadata["processed_at"] = datetime.now().isoformat()
        #     processed_docs.append(doc)
        # return processed_docs

        query = inputs.get("input") or inputs.get("query")
        return self.retriever.get_relevant_documents(query)
    #
    # def _call(self, inputs, run_manager=None, **kwargs):
    #     # Получаем документы
    #     docs = self._get_docs(inputs, run_manager=run_manager)
    #     if not docs or docs[0].page_content == "Пожалуйста, задайте вопрос иначе или уточните его.":
    #         return {"result": "Пожалуйста, задайте вопрос иначе или уточните его.", "source_documents": docs}
    #
    #     # Формируем message_content как в answer_index
    #     message_content = re.sub(
    #         r'\n{2}', ' ',
    #         '\n '.join([
    #             f'\nОтрывок документа №{i + 1}\n=====================\n{doc.page_content}\n'
    #             for i, doc in enumerate(docs)
    #         ])
    #     )
    #     system_prompt = inputs.get("system_prompt", self.retriever.system_prompt)
    #     # Формируем полный промпт
    #     prompt = PromptTemplate(
    #         input_variables=["system_prompt", "context", "input"],
    #         template="SYSTEM: {system_prompt}\n\nCONTEXT: {context}\n\nQuestion: {input}"
    #     )
    #     formatted_prompt = prompt.format(
    #         system_prompt=self.retriever.system_prompt,
    #         context=message_content,
    #         input=inputs["query"]
    #     )
    #     # Передаем кастомный промпт в StuffDocumentsChain
    #     llm_chain = self.combine_documents_chain.llm_chain
    #     llm_chain.prompt = prompt
    #     result = self.combine_documents_chain(
    #         {"context": message_content, "question": inputs["query"], "system_prompt": system_prompt})
    #     return {"result": result["output_text"], "source_documents": docs}
    #
    #     # # Вызываем LLM
    #     # result = self.llm.invoke(formatted_prompt)
    #     # return {"result": result.content, "source_documents": docs}

    def _call(self, inputs: dict, run_manager: CallbackManagerForChainRun = None) -> Dict[str, Any]:
        query = inputs.get("input") or inputs.get("query")
        system_prompt = inputs.get("system_prompt")

        # 1. Получаем документы
        docs = self._get_docs(inputs, run_manager)

        # 2. Собираем контекстные данные
        new_inputs = {
            "input_documents": docs,
            "input": query,
            "system_prompt": system_prompt or "",
        }

        result = self.combine_documents_chain.invoke(new_inputs, run_manager=run_manager)
        return {
            "result": result["output_text"],  # <-- правильно отдаем result
            "source_documents": docs  # <-- важно, даже если docs пустой
        }


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
    # for storage_set in storage_sets:
    #     for storage in storage_set.all():
    #         # Формируем путь к FAISS индексу
    #         faiss_dir = os.path.join(
    #             BASE_DIR, "media", "kb", str(kb.pk), "embedding_stores",
    #             f"{storage.__class__.__name__}_id_{storage.pk}_embedding_store",
    #             f"{embedding_engine.name}_faiss_index_db"
    #         )
    #         try:
    #             db_index = get_vectorstore(
    #                 path=faiss_dir,
    #                 embeddings=embeddings_model
    #             )
    #         except Exception as e:
    #             print(f"Ошибка загрузки векторного хранилища для {storage.__class__.__name__} id {storage.pk}: {str(e)}")
    #             db_index = None
    #
    #         if db_index:
    #             print(db_index)
    #             print(storage.name)
    #             print(storage.description)
    #             retriever = CustomRetriever(
    #                 db_index=db_index,
    #                 system_prompt=kb.system_instruction
    #             )
    #             retriever_infos.append({
    #                 "name": f"retriever_{storage.__class__.__name__}_{storage.pk}",
    #                 "retriever": retriever,
    #                 "description": f"{storage.name} — {storage.description}",
    #                 "chain_type": CustomRetrievalQA,
    #                 "prompt": PromptTemplate(
    #                             input_variables=["system_prompt", "input"],
    #                             template="SYSTEM: {system_prompt}\n\nCONTEXT: {context}\n\nQuestion: {input}"
    #                         ),
    #                 "chain_type_kwargs": {
    #                     "return_source_documents": True
    #                 }
    #             })
    #             if storage.default_retriever:
    #                 default_retriever = retriever
    # if not retriever_infos:
    #     raise ValueError("Не удалось создать ретриверы для данной базы знаний.")
    # if not default_retriever:
    #     default_retriever = retriever_infos[0]
    #
    # print("retriever_infos:", [info["name"] for info in retriever_infos])
    #
    #
    #
    # # retriever = db_simblie_merged.as_retriever(
    # #     search_type="similarity",
    # #     search_kwargs={"k": 4}
    # # )
    # #
    # # template = """Отвечайте на вопросы только на основе следующего контекста:
    # #
    # # {context}
    # #
    # # Вопрос: {question}
    # # """
    # # prompt = ChatPromptTemplate.from_template(template)
    # # model = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)
    # #
    # # def format_docs(docs):
    # #     return "\n\n".join([d.page_content for d in docs])
    # #
    # # chain = (
    # #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # #         | prompt
    # #         | model
    # #         | StrOutputParser()
    # # )
    #
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    # # default_chain = CustomRetrievalQA(
    # #     llm=llm,
    # #     retriever=default_retriever,
    # #     return_source_documents=True,
    # #     chain_type_kwargs={
    # #         "prompt": PromptTemplate(
    # #             input_variables=["system_prompt", "context", "input"],
    # #             template="SYSTEM: {system_prompt}\n\nCONTEXT: {context}\n\nQuestion: {input}"
    # #         )
    # #     }
    # # )
    # prompt = PromptTemplate(
    #     input_variables=["system_prompt", "context", "input"],
    #     template="SYSTEM: {system_prompt}\n\nCONTEXT: {context}\n\nQuestion: {input}"
    # )
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    # combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
    #
    # default_chain = CustomRetrievalQA(
    #     retriever=default_retriever,
    #     combine_documents_chain=combine_documents_chain,
    #     return_source_documents=True
    # )
    #
    # multi_chain = MultiRetrievalQAChain.from_retrievers(
    #     llm=llm,
    #     retriever_infos=retriever_infos,
    #     default_chain=default_chain,
    #     verbose=True
    # )
    # return multi_chain
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
                system_prompt=kb.system_instruction
            )

            prompt = PromptTemplate(
                input_variables=["system_prompt", "context", "input"],
                template="SYSTEM: {system_prompt}\n\nCONTEXT: {context}\n\nQuestion: {input}"
            )
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
        f"{name}: {destination_chains[name].retriever.system_prompt[:50]}..."
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