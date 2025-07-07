import os
import pickle

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import json

from test_db_zone.aux_module import create_vector_db, VectorDbType


def create_documents_from_chuncks(filename):

    with open("./chunk_example.json", encoding='utf-8') as dataset_file:
        dataset = json.load(dataset_file)

    source_chunks = []

    for chunk in dataset:
        source_chunks.append(
            Document(
                page_content=chunk['text'],
                metadata={
                    'file': chunk['file'],
                    'chunk_index': chunk['chunk_index'],
                    'id': chunk['id'],
                    'prev': chunk['prev'],
                    'next': chunk['next'],
                    'header_l1': chunk['metadata'].get("Header 1", None),
                    'header_l2': chunk['metadata'].get("Header 2", None),
                    'url': chunk['metadata']['url']
                }
            )
        )
    return source_chunks


def load_documents_from_file(filename):
    # Определяем путь к файлу
    file_path = os.path.join(filename)

    # Проверка, существует ли файл
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден!")
        return []

    # Открываем файл для чтения в бинарном режиме
    with open(file_path, 'rb') as f:
        # Загружаем (десериализуем) список документов из файла
        all_documents = pickle.load(f)
    print(f"Документы загружены из файла: {file_path}")
    return all_documents

from langchain_openai import OpenAIEmbeddings

if __name__ == "__main__":
    documents = load_documents_from_file("chunk.pickle")
    print("start loading vector DB")
    # embedding = HuggingFaceEmbeddings(
    #     model_name="ai-forever/sbert_large_nlu_ru",
    #     encode_kwargs={"normalize_embeddings": True, "batch_size": 8}  # Меньший батч
    # )
    # embedding = HuggingFaceEmbeddings(
    #     model_name="intfloat/multilingual-e5-base",
    #     encode_kwargs={"normalize_embeddings": True, "batch_size": 8}  # Меньший батч
    # )
    embedding = HuggingFaceEmbeddings(
        model_name="ai-forever/FRIDA",
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8}
    )

    create_vector_db(
        db_type=VectorDbType.faissdb,
        chunks=documents,
        embedding=embedding,
        # embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    print("End loading vector DB")
