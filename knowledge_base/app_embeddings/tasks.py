import json
import logging
import os
import re
import time
from datetime import datetime

import requests
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.contrib.auth import get_user_model
from django.db.models import OuterRef, Subquery, Q
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from app_chat.models import ChatSession, ChatMessage
from app_chunks.models import Chunk, ChunkStatus
from app_core.models import KnowledgeBase
from app_sources.content_models import URLContent, ContentStatus, CleanedContent, RawContent
from app_sources.source_models import URL, SourceStatus, NetworkDocument
from app_sources.storage_models import WebSite, CloudStorage, LocalStorage, URLBatch
from knowledge_base.settings import BASE_DIR
from telegram_bot.bot_config import KB_AI_API_KEY
from utils.setup_logger import setup_logger
from .models import Embedding, EmbeddingsReport
# from .services.embedding_config import system_instruction
from .services.embedding_store import load_embedding, get_vectorstore
from .services.retrieval_engine import answer_index

User = get_user_model()
logger = setup_logger(name=__file__, log_dir="logs/embeddings", log_file="embeddings.log")


@shared_task(bind=True)
def universal_create_vectors_task(self, author_pk, report_pk):
    """
    Celery задача для векторизации чанков, связанных с WebSite, с поддержкой прогресс-бара.
    """

    progress_recorder = ProgressRecorder(self)

    report = (EmbeddingsReport.objects
              .select_related("site__kb", "batch__kb", "cloud_storage__kb", "local_storage__kb")
              .get(pk=report_pk))
    logger.info(f"EmbeddingsReport [id {report.pk}] старт {__file__}")
    storage = (
            report.site or
            report.batch or
            report.cloud_storage or
            report.local_storage
    )

    if not storage:
        raise ValueError(
            f"Ошибка связанного хранилища для отчета векторизации EmbeddingsReport [id {report.pk}]"
        )

    kb = storage.kb
    logger.info(f"EmbeddingsReport [id {report.pk}] База знаний: {kb.name} (id {kb.pk})")

    # Проверяем, есть ли связанный EmbeddingEngine
    if not kb.engine:
        raise ValueError(f"EmbeddingsReport [id {report.pk}] Для базы знаний не указан движок эмбеддинга")

    embedding_engine = kb.engine
    embeddings_model_name = embedding_engine.model_name

    try:
        embeddings_model = load_embedding(embeddings_model_name)
    except Exception as e:
        raise ValueError(f"EmbeddingsReport [id {report.pk}] Ошибка загрузки модели {embeddings_model_name}: {str(e)}")

    chunks = Chunk.objects.none()

    if isinstance(storage, CloudStorage):
        logger.info("EmbeddingsReport [id {report.pk}] Storage is CloudStorage: %s", storage)
        chunks = Chunk.objects.filter(
            Q(cleaned_content__network_document__storage=storage) |
            Q(raw_content__network_document__storage=storage),
            embedding__isnull=True
        ).exclude(status=ChunkStatus.ERROR.value).distinct()

    elif isinstance(storage, LocalStorage):
        logger.info("EmbeddingsReport [id {report.pk}] Storage is LocalStorage: %s", storage)
        chunks = Chunk.objects.filter(
            Q(cleaned_content__local_document__storage=storage) |
            Q(raw_content__local_document__storage=storage),
            embedding__isnull=True
        ).exclude(status=ChunkStatus.ERROR.value).distinct()

    elif isinstance(storage, WebSite):
        logger.info("EmbeddingsReport [id {report.pk}] Storage is WebSite: %s", storage)
        chunks = Chunk.objects.filter(
            url_content__url__site=storage,
            embedding__isnull=True
        ).exclude(status=ChunkStatus.ERROR.value).distinct()

    elif isinstance(storage, URLBatch):
        logger.info("EmbeddingsReport [id {report.pk}] Storage is URLBatch: %s", storage)
        # оставляем chunks пустым

    else:
        logger.warning("EmbeddingsReport [id {report.pk}] Неизвестный тип storage: %s", type(storage))
        # chunks остаётся пустым

    # Проверка наличия чанков
    if not chunks.exists():
        raise ValueError("EmbeddingsReport [id {report.pk}] Чанки для обработки не найдены")

    batch_size = 10
    total_chunks = chunks.count()
    processed_chunks = 0
    new_embeddings = []
    vector_ids = []

    # Инициализация или загрузка FAISS индекса
    faiss_dir = os.path.join(BASE_DIR, "media", "kb", str(kb.pk), "embedding_stores",
                             f"{storage.__class__.__name__}_id_{storage.pk}_embedding_store",
                             f"{embedding_engine.name}_faiss_index_db")

    if not os.path.exists(faiss_dir):
        os.makedirs(faiss_dir)
    try:
        db_index = get_vectorstore(
            path=faiss_dir,
            embeddings=embeddings_model
        )
    except Exception as e:
        logger.error(f"EmbeddingsReport [id {report.pk}]", e)
        db_index = FAISS.from_documents([Document(page_content='', metadata={})], embeddings_model)

    seen_chunks = set()

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
        logger.info(f"EmbeddingsReport [id {report.pk}] {batch_chunk_ids=}")
        logger.info(f"EmbeddingsReport [id {report.pk}] {batch_vector_ids=}")

        # Создаем записи Embedding
        for chunk_id, vector_id in zip(batch_chunk_ids, batch_vector_ids):

            if chunk_id in seen_chunks:
                logger.error(
                    f"[DUPLICATE ADD] chunk_id={chunk_id} уже добавлен ранее в new_embeddings."
                )
                logger.error(f"{batch_chunk_ids=}")
                logger.error(f"{batch_vector_ids=}")
            else:
                seen_chunks.add(chunk_id)
            # chunk = Chunk.objects.get(id=chunk_id)
            # if not Embedding.objects.filter(chunk=chunk, embedding_engine=embedding_engine).exists():
            new_embeddings.append(
                Embedding(
                    chunk_id=chunk_id,
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
    new_embeddings = Embedding.objects.bulk_create(new_embeddings)

    # Сохраняем FAISS индекс
    db_index.save_local(folder_path=faiss_dir,  # путь к папке (path)
                        index_name="index")  # имя для индексной базы (index_name)

    chunk_ids = [emb.chunk_id for emb in new_embeddings]
    Chunk.objects.filter(id__in=chunk_ids).update(status=ChunkStatus.ACTIVE.value)

    if isinstance(storage, CloudStorage):
        cleaned_content_ids = (
            Chunk.objects
            .filter(embedding__in=new_embeddings)
            .values_list("cleaned_content_id", flat=True)
            .distinct()
        )
        CleanedContent.objects.filter(id__in=cleaned_content_ids).update(status=ContentStatus.ACTIVE.value)

        raw_content_ids = (
            Chunk.objects
            .filter(embedding__in=new_embeddings)
            .values_list("raw_content_id", flat=True)
            .distinct()
        )
        RawContent.objects.filter(id__in=raw_content_ids).update(status=ContentStatus.ACTIVE.value)

        network_documents_ids_for_raw = (
            RawContent.objects
            .filter(id__in=raw_content_ids, )
            .values_list("network_document_id", flat=True)
            .distinct()
        )
        network_documents_ids_for_cleaned = (
            CleanedContent.objects
            .filter(id__in=cleaned_content_ids, )
            .values_list("network_document_id", flat=True)
            .distinct()
        )
        all_doc_ids = set(network_documents_ids_for_raw) | set(network_documents_ids_for_cleaned)

        NetworkDocument.objects.filter(id__in=all_doc_ids).update(status=SourceStatus.ACTIVE.value)
        logger.info(f"Activated {len(all_doc_ids)} NetworkDocuments for {storage}")

    elif isinstance(storage, LocalStorage):
        pass
    elif isinstance(storage, WebSite):
        url_content_ids = (
            Chunk.objects
            .filter(embedding__in=new_embeddings)
            .values_list("url_content_id", flat=True)
            .distinct()
        )
        URLContent.objects.filter(id__in=url_content_ids).update(status=ContentStatus.ACTIVE.value)

        url_ids = (
            URLContent.objects
            .filter(id__in=url_content_ids)
            .values_list("url_id", flat=True)
            .distinct()
        )
        URL.objects.filter(id__in=url_ids).update(status=SourceStatus.ACTIVE.value)

    elif isinstance(storage, URLBatch):
        pass

    return f"Обработано {len(new_embeddings)} чанков, создано {len(vector_ids)} эмбеддингов"


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

    chunks = (
        Chunk.objects
        .filter(
            url_content__url__site=website,
            embedding__isnull=True
        )
        .exclude(status=ChunkStatus.ERROR.value).distinct()
    )

    if not chunks.exists():
        raise ValueError("Чанки для обработки не найдены")

    batch_size = 10
    total_chunks = chunks.count()
    processed_chunks = 0
    new_embeddings = []
    vector_ids = []

    # Инициализация или загрузка FAISS индекса
    faiss_dir = os.path.join(BASE_DIR, "media", "kb", str(kb.pk), "embedding_stores",
                             f"{website.__class__.__name__}_id_{website.pk}_embedding_store",
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
    new_embedding_ids = Embedding.objects.bulk_create(new_embeddings)

    # Сохраняем FAISS индекс
    db_index.save_local(folder_path=faiss_dir,  # путь к папке (path)
                        index_name="index")  # имя для индексной базы (index_name)

    # Получаем список chunk.id из сохранённых Embedding
    chunk_ids = [embedding.chunk_id for embedding in new_embeddings]

    # Обновляем статус чанков
    Chunk.objects.filter(id__in=chunk_ids).update(status=ChunkStatus.ACTIVE.value)

    # Получаем id связанных URLContent и обновляем их статус
    url_content_ids = (
        Chunk.objects
        .filter(id__in=chunk_ids)
        .values_list("url_content_id", flat=True)
        .distinct()
    )
    URLContent.objects.filter(id__in=url_content_ids).update(status=ContentStatus.ACTIVE.value)

    # Получаем id связанных URL и обновляем их статус
    url_ids = (
        URLContent.objects
        .filter(id__in=url_content_ids)
        .values_list("url_id", flat=True)
        .distinct()
    )
    URL.objects.filter(id__in=url_ids).update(status=SourceStatus.ACTIVE.value)

    return f"Обработано {len(chunk_ids)} чанков, создано {len(new_embeddings)} эмбеддингов"




@shared_task(bind=True, soft_time_limit=3700, time_limit=3800)
def test_task(self, steps=60, sleep_per_step=60.0):
    """
    Тестовая Celery-задача, работающая ~1 час.

    Параметры:
    - steps (int): количество этапов.
    - sleep_per_step (float): продолжительность "работы" на каждом этапе (в секундах).

    Логирует время начала и окончания каждого этапа.
    """
    progress_recorder = ProgressRecorder(self)

    logger.info(f"[test_task] Задача стартовала: steps={steps}, sleep_per_step={sleep_per_step:.2f}s")

    for step in range(1, steps + 1):
        start_time = datetime.now()
        logger.info(f"[test_task] Этап {step}/{steps} стартовал в {start_time.strftime('%H:%M:%S')}")

        try:
            # Имитация работы
            time.sleep(sleep_per_step)
        except Exception as e:
            logger.exception(f"[test_task] Ошибка на этапе {step}: {e}")
            raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            f"[test_task] Этап {step} завершён в {end_time.strftime('%H:%M:%S')} (длительность: {duration:.2f} сек)")

        # Обновление прогресса
        progress_recorder.set_progress(step, steps)

    logger.info("[test_task] Задача завершена успешно.")
