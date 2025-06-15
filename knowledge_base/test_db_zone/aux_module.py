from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from enum import Enum
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from uuid import uuid4
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
# from flashrank import Ranker


class VectorDbType(str, Enum):
    chromadb = "Chroma"
    faissdb = "FAISS"
    qdrant = "qdrant"
    qdrant_bm25 = "qdrant_bm25"
    qdrant_dense_docker = "qdrant_dense_docker"


class RetrieverType(str, Enum):
    only_chromadb = "only_chromadb"
    only_faissdb = "only_faissdb"
    only_qdrant = "only_qdrant"
    only_bm25 = "only_bm25"
    chroma_bm25 = "chroma_bm25"
    faissdb_bm25 = "faissdb_bm25"
    qdrant_bm25 = "qdrant_bm25"
    chroma_bm25_rerank = "chroma_bm25_rerank"
    only_qdrant_docker = "only_qdrant_docker"


def create_vector_db(
        db_type: VectorDbType,
        chunks: list[Document],
        embedding: Embeddings
):
    vectordb_index = None
    if db_type == VectorDbType.chromadb:
        vectordb_index = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory='./chroma_index_db'
        )
    elif db_type == VectorDbType.faissdb:
        vectordb_index = FAISS.from_documents(
            documents=chunks,
            embedding=embedding
        )
        vectordb_index.save_local('e5_faiss_index_db')
    elif db_type == VectorDbType.qdrant:
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        client = QdrantClient(path="./langchain_qdrant")
        client.create_collection(
            collection_name="dense_db",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        vectordb_index = QdrantVectorStore(
            client=client,
            collection_name="dense_db",
            embedding=embedding,
            retrieval_mode=RetrievalMode.DENSE,
        )

        vectordb_index.add_documents(documents=chunks, ids=uuids)

    elif db_type == VectorDbType.qdrant_bm25:
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        client = QdrantClient(path="./langchain_qdrant")
        client.create_collection(
            collection_name="hybrid_db",
            vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        vectordb_index = QdrantVectorStore(
            client=client,
            collection_name="hybrid_db",
            embedding=embedding,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        vectordb_index.add_documents(documents=chunks, ids=uuids)
    elif db_type == VectorDbType.qdrant_dense_docker:
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        client = QdrantClient(url="http://localhost:6333")
        client.create_collection(
            collection_name="qdrant_dense_docker",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        vectordb_index = QdrantVectorStore(
            client=client,
            collection_name="qdrant_dense_docker",
            embedding=embedding,
            retrieval_mode=RetrievalMode.DENSE,
        )

        vectordb_index.add_documents(documents=chunks, ids=uuids)

    return vectordb_index


def load_and_init_retrievers(
        retriver_type: RetrieverType,
        embedding: Embeddings,
        number_or_return_docs: int,
        chunks: list[Document] = None
):
    retriever = None
    if retriver_type == RetrieverType.only_faissdb:
        faiss_index_db = FAISS.load_local(
            folder_path='faiss_index_db',
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        retriever = faiss_index_db.as_retriever(search_kwargs={"k": number_or_return_docs})
    elif retriver_type == RetrieverType.only_chromadb:
        chroma_index_db = Chroma(
            persist_directory='./chroma_index_db',
            embedding_function=embedding,
        )
        retriever = chroma_index_db.as_retriever(search_kwargs={"k": number_or_return_docs})
    elif retriver_type == RetrieverType.only_bm25:
        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = number_or_return_docs
    elif retriver_type == RetrieverType.faissdb_bm25:
        faiss_index_db = FAISS.load_local(
            folder_path='faiss_index_db',
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        faiss_retriever = faiss_index_db.as_retriever(search_kwargs={"k": number_or_return_docs})

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = number_or_return_docs

        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
    elif retriver_type == RetrieverType.chroma_bm25:
        chroma_index_db = Chroma(
            persist_directory='./chroma_index_db',
            embedding_function=embedding,
        )
        chroma_retriever = chroma_index_db.as_retriever(search_kwargs={"k": number_or_return_docs})

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = number_or_return_docs

        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
        )
    elif retriver_type == RetrieverType.only_qdrant:
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embedding,
            retrieval_mode=RetrievalMode.DENSE,
            path="./langchain_qdrant",
            collection_name="dense_db",
        )
        retriever = qdrant.as_retriever(search_kwargs={"k": number_or_return_docs})
    elif retriver_type == RetrieverType.qdrant_bm25:
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        qdrant = QdrantVectorStore.from_existing_collection(
            collection_name="hybrid_db",
            embedding=embedding,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
            path="./langchain_qdrant",
        )
        retriever = qdrant.as_retriever(search_kwargs={"k": number_or_return_docs})
    elif retriver_type == RetrieverType.chroma_bm25_rerank:
        # ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

        chroma_index_db = Chroma(
            persist_directory='./chroma_index_db',
            embedding_function=embedding,
        )
        chroma_retriever = chroma_index_db.as_retriever(search_kwargs={"k": number_or_return_docs})

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = number_or_return_docs

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
        )
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=number_or_return_docs)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )
    elif retriver_type == RetrieverType.only_qdrant_docker:
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embedding,
            retrieval_mode=RetrievalMode.DENSE,
            url="http://localhost:6333",
            collection_name="qdrant_dense_docker",
        )
        retriever = qdrant.as_retriever(search_kwargs={"k": number_or_return_docs})
    return retriever
