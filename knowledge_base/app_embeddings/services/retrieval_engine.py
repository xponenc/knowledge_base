# retrieval_engine.py
import re
from openai import OpenAI

from app_embeddings.services.embedding_config import FAISS_THRESHOLD, TOP_N
# from app_embeddings.services.embedding_store import RERANKER, VECTORSTORE


# def rerank_documents(query, docs_with_scores, reranker=RERANKER, threshold=FAISS_THRESHOLD, top_n=TOP_N):
def rerank_documents(query, docs_with_scores, reranker, threshold=FAISS_THRESHOLD, top_n=TOP_N):
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


def answer_index(system, query, verbose=False):
    docs_with_scores = VECTORSTORE.similarity_search_with_score(query, k=10)
    if verbose:
        print("Вывод docs_with_scores ========= ")
        for doc in docs_with_scores:
            print("\n", doc)
        print("========= \n\n\n")
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
