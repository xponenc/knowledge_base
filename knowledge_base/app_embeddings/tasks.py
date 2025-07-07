import logging
import os

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.contrib.auth import get_user_model
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from app_chunks.models import Chunk, ChunkStatus
from app_sources.storage_models import WebSite
from knowledge_base.settings import BASE_DIR
from .models import Embedding
from .services.embedding_store import load_embedding, get_vectorstore

User = get_user_model()
logger = logging.getLogger(__name__)

@shared_task(bind=True)
def create_vectors_task(self, website_id, author_pk, report_pk):
    """
    Celery задача для векторизации чанков, связанных с WebSite, с поддержкой прогресс-бара.
    """
    progress_recorder = ProgressRecorder(self)
    website = WebSite.objects.select_related("kb").get(id=website_id)
    # user = User.objects.get(id=author_pk)

    kb = website.kb

    # Проверяем, есть ли связанный EmbeddingEngine
    if not kb.engine:
        raise ValueError("Для базы знаний не указан движок эмбеддинга")

    embedding_engine = kb.engine
    embeddings_model_name = embedding_engine.model_name

    try:
        embeddings_model = load_embedding(embeddings_model_name)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки модели {embeddings_model_name}: {str(e)}")

    chunks = (Chunk.objects
              .filter(
                url_content__url__site=website,
                embedding__isnull=True
                )
              .exclude(status=ChunkStatus.ERROR.value)
             )

    if not chunks.exists():
        raise ValueError("Чанки для обработки не найдены")

    batch_size = 900
    total_chunks = chunks.count()
    processed_chunks = 0
    new_embeddings = []
    vector_ids = []

    # Инициализация или загрузка FAISS индекса
    faiss_dir = os.path.join(BASE_DIR, "media", "kb", str(kb.pk), "embedding_store",
                             f"{embedding_engine.name}_faiss_index_db")

    if not os.path.exists(faiss_dir):
        os.makedirs(faiss_dir)
    try:
        db_index = get_vectorstore(
            path=faiss_dir,
            embeddings=embeddings_model
        )
    except Exception as e:
        logger.error(e)
        db_index = FAISS.from_documents([Document(page_content='', metadata={})], embeddings_model)

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_texts = []
        batch_metadatas = []
        batch_chunk_ids = []

        for chunk in batch_chunks:
            if not chunk.page_content:
                continue
            # Создаем Document в цикле
            metadata = chunk.metadata or {}
            metadata["chunk_id"] = chunk.id
            batch_texts.append(chunk.page_content)
            batch_metadatas.append(metadata)
            batch_chunk_ids.append(chunk.id)

        if not batch_texts:
            continue

        # Создаем вектора для батча
        batch_vector_ids = db_index.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas
        )

        # Создаем записи Embedding
        for chunk_id, vector_id in zip(batch_chunk_ids, batch_vector_ids):
            chunk = Chunk.objects.get(id=chunk_id)
            if not Embedding.objects.filter(chunk=chunk, embedding_engine=embedding_engine).exists():
                new_embeddings.append(
                    Embedding(
                        chunk=chunk,
                        embedding_engine=embedding_engine,
                        vector_id=vector_id,
                        author_id=author_pk,
                        report_id=report_pk,
                    )
                )

        # Обновляем прогресс
        processed_chunks += len(batch_texts)
        progress_recorder.set_progress(
            processed_chunks,
            total_chunks,
            description=f"Обработано {processed_chunks}/{total_chunks} чанков"
        )

        # Сохраняем записи в базе данных
    Embedding.objects.bulk_create(new_embeddings)

    # Сохраняем FAISS индекс
    db_index.save_local(folder_path=faiss_dir,  # путь к папке (path)
                        index_name="index")  # имя для индексной базы (index_name)

    return f"Обработано {len(new_embeddings)} чанков, создано {len(vector_ids)} эмбеддингов"
