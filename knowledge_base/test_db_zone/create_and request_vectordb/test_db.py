from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Если хочешь использовать invoke вместо get_relevant_documents
from langchain_core.runnables import Runnable
from langchain_openai import OpenAIEmbeddings

FAISS_DB_PATH = "./faiss_index_db"
OAI_FAISS_DB_PATH = "./openai_faiss_index_db"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Загрузка базы FAISS
vector_db = FAISS.load_local(FAISS_DB_PATH, embeddings=embedding, index_name="index",
                             allow_dangerous_deserialization=True, )

# Создание ретривера как Runnable
retriever: Runnable = vector_db.as_retriever(search_kwargs={"k": 5})

# Запрос пользователя
query = "Хочу пройти курс по охране окружающей среды для проектирования"

# Новый способ поиска — через invoke
results = retriever.invoke(query)

# Выводим найденные документы
for i, doc in enumerate(results, 1):
    print(f"--- Документ {i} ---")
    print(doc.page_content)

print("OpenAIEmbeddings")

# Загрузка базы FAISS
oai_vector_db = FAISS.load_local(OAI_FAISS_DB_PATH, embeddings=OpenAIEmbeddings(
    openai_api_key=""),
                                 index_name="index", allow_dangerous_deserialization=True, )

# Новый способ поиска — через invoke
results = oai_vector_db.similarity_search(query, k=5)

# Выводим найденные документы
for i, doc in enumerate(results, 1):
    print(f"--- Документ {i} ---")
    print(doc.page_content)

print("ai-forever/sbert_large_nlu_ru")

SBER_FAISS_DB_PATH = "./sber_faiss_index_db"
embedding = HuggingFaceEmbeddings(
    model_name="ai-forever/sbert_large_nlu_ru",
    encode_kwargs={"normalize_embeddings": True, "batch_size": 8}  # Меньший батч
)
sber_vector_db = FAISS.load_local(SBER_FAISS_DB_PATH, embeddings=embedding,
                                  index_name="index", allow_dangerous_deserialization=True, )

results = sber_vector_db.similarity_search(query, k=5)

# Выводим найденные документы
for i, doc in enumerate(results, 1):
    print(f"--- Документ {i} ---")
    print(doc.page_content)

print("intfloat/multilingual-e5-base")

e5_FAISS_DB_PATH = "./e5_faiss_index_db"
embedding = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={"normalize_embeddings": True, "batch_size": 8}  # Меньший батч
)
e5_vector_db = FAISS.load_local(e5_FAISS_DB_PATH, embeddings=embedding,
                                index_name="index", allow_dangerous_deserialization=True, )

results = e5_vector_db.similarity_search(query, k=5)

# Выводим найденные документы
for i, doc in enumerate(results, 1):
    print(f"--- Документ {i} ---")
    print(doc.page_content)
