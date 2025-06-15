from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import json
from langchain_huggingface import HuggingFaceEmbeddings

from test_db_zone.aux_module import load_and_init_retrievers, RetrieverType

# раскомментировать блок, если используется bm25 в faiss и chromadb
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

retriever = load_and_init_retrievers(
    retriver_type=RetrieverType.only_qdrant_docker,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
    number_or_return_docs=5,
    chunks=source_chunks # раскомментить в случае bm25 для faiss и chromadb
)

while True:
    query = input("Введите запрос: ")
    if query == "exit":
        break
    print(retriever.invoke(query))

