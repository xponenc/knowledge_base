import json
import logging
import os

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.contrib.auth import get_user_model
from django.db.models import OuterRef, Subquery, Q
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from app_chunks.models import Chunk, ChunkStatus
from app_sources.content_models import URLContent, ContentStatus, CleanedContent, RawContent
from app_sources.source_models import URL, SourceStatus, NetworkDocument
from app_sources.storage_models import WebSite, CloudStorage, LocalStorage, URLBatch
from knowledge_base.settings import BASE_DIR
from .models import Embedding, EmbeddingsReport
from .services.embedding_config import system_instruction
from .services.embedding_store import load_embedding, get_vectorstore
from .services.retrieval_engine import answer_index

User = get_user_model()
logger = logging.getLogger(__name__)


@shared_task(bind=True)
def universal_create_vectors_task(self, author_pk, report_pk):
    """
    Celery задача для векторизации чанков, связанных с WebSite, с поддержкой прогресс-бара.
    """
    progress_recorder = ProgressRecorder(self)

    report = (EmbeddingsReport.objects
              .select_related("site__kb", "batch__kb", "cloud_storage__kb", "local_storage__kb")
              .get(pk=report_pk))

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

    # Проверяем, есть ли связанный EmbeddingEngine
    if not kb.engine:
        raise ValueError("Для базы знаний не указан движок эмбеддинга")

    embedding_engine = kb.engine
    embeddings_model_name = embedding_engine.model_name

    try:
        embeddings_model = load_embedding(embeddings_model_name)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки модели {embeddings_model_name}: {str(e)}")

    chunks = Chunk.objects.none()

    if isinstance(storage, CloudStorage):
        logger.info("Storage is CloudStorage: %s", storage)
        chunks = Chunk.objects.filter(
            Q(cleaned_content__network_document__storage=storage) |
            Q(raw_content__network_document__storage=storage),
            embedding__isnull=True
        ).exclude(status=ChunkStatus.ERROR.value)

    elif isinstance(storage, LocalStorage):
        logger.info("Storage is LocalStorage: %s", storage)
        chunks = Chunk.objects.filter(
            Q(cleaned_content__local_document__storage=storage) |
            Q(raw_content__local_document__storage=storage),
            embedding__isnull=True
        ).exclude(status=ChunkStatus.ERROR.value)

    elif isinstance(storage, WebSite):
        logger.info("Storage is WebSite: %s", storage)
        chunks = Chunk.objects.filter(
            url_content__url__site=storage,
            embedding__isnull=True
        ).exclude(status=ChunkStatus.ERROR.value)

    elif isinstance(storage, URLBatch):
        logger.info("Storage is URLBatch: %s", storage)
        # оставляем chunks пустым

    else:
        logger.warning("Неизвестный тип storage: %s", type(storage))
        # chunks остаётся пустым

    # Проверка наличия чанков
    if not chunks.exists():
        raise ValueError("Чанки для обработки не найдены")

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

    chunks = (Chunk.objects
              .filter(
        url_content__url__site=website,
        embedding__isnull=True
    )
              .exclude(status=ChunkStatus.ERROR.value)
              )

    if not chunks.exists():
        raise ValueError("Чанки для обработки не найдены")

    batch_size = 10
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
    new_embedding_ids = Embedding.objects.bulk_create(new_embeddings)

    # Сохраняем FAISS индекс
    db_index.save_local(folder_path=faiss_dir,  # путь к папке (path)
                        index_name="index")  # имя для индексной базы (index_name)

    Chunk.objects.filter(embedding_id__in=new_embedding_ids).update(status=ChunkStatus.ACTIVE.value)

    url_content_ids = (
        Chunk.objects
        .filter(embedding_id__in=new_embedding_ids)
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

    return f"Обработано {len(new_embeddings)} чанков, создано {len(vector_ids)} эмбеддингов"


def create_test_data(content: str, ai_prompt: str):
    system_prompt = """
    Ты - опытный эксперт и специалист по настройке Ai-помощников.
    Сформируй вопрос и ответ для тестирования ответа ai по предоставленному тексту и инструкции для тестируемого ai.
    Вопрос сформируй в формате максимально приближенному к разговорному стилю общения или стилю общения в чатах.
    Ответ должен быть строго в формате json, соответствующем Python-словарю:
    {"answer": "...", "question": "..."}

    Не добавляй никаких пояснений, текста до или после — только json.
    """

    client = OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Ответь на вопрос. Текст для теста: {content}\n\nИнструкция тестируемого AI: \n{ai_prompt}"}
    ]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    answer = completion.choices[0].message.content
    return answer


def get_model_response_evaluation(
        content: str,
        benchmark_question: str,
        benchmark_answer: str,
        ai_answer: str,
        ai_prompt: str):
    system_prompt = """
    Ты - опытный эксперт и специалист по настройке Ai-помощников.
    Тебе будут предоставлены следующие данные:
    Исходный текст
    Эталонный вопрос по тексту
    Эталонный ответ на "Эталонный вопрос по тексту"
    Ответ тестируемого AI на "Эталонный вопрос по тексту"
    Инструкция для тестируемого AI
    На основании этой информации оцени Ответ тестируемого AI и выдай ответ
    Ответ должен быть строго в формате json, соответствующем Python-словарю:
    {"answer_score": "оценка Ответ тестируемого AI на соответствие Эталонному ответу от -2 до 2, по правилам:
    2	Полное соответствие, вся информация присутствует, релевантна
    1	Основное соответствие, большая часть информации присутствует, но есть мелкие упущения или незначительные нерелевантные данные
    0	Частичное соответствие, сложно оценить, некоторая релевантная информация есть, но много пропущено или не относится к вопросу.
    -1	Основное несоответствие, мало релевантной информации, много нерелевантной или вводящей в заблуждение
    -2	Полное несоответствие, нет релевантной информации или данные ошибочны.
    ",
     "prompt_score": "оценка Ответ тестируемого AI на соответствие от Инструкция для тестируемого AI 0 до 10",
     "answer_resume": "краткое резюме по answer_score",
     "prompt_resume": "краткое резюме по prompt_score",
     }
    Не добавляй никаких пояснений, текста до или после — только json.
    """

    client = OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Ответь на вопрос. "
                    f"Исходный текст: {content}\n\n"
                    f"Эталонный вопрос по тексту: {benchmark_question}\n\n"
                    f"Эталонный ответ на Эталонный вопрос по тексту: {benchmark_answer}\n\n"
                    f"Ответ тестируемого AI на Эталонный вопрос по тексту: {ai_answer}\n\n"
                    f"Инструкция тестируемого AI: \n{ai_prompt}"}
    ]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    answer = completion.choices[0].message.content
    return answer


@shared_task(bind=True)
def test_model_answer(self,
                      test_urls: list[str],
                      ):
    """
    """

    latest_content_qs = URLContent.objects.filter(url=OuterRef('pk')).order_by('-created_at')

    urls = URL.objects.filter(url__in=test_urls).annotate(
        latest_content_body=Subquery(latest_content_qs.values('body')[:1])
    )

    total_counter = len(urls)
    if total_counter == 0:
        return "Обработка завершена: документы не найдены"

    # Инициализация прогресса
    progress_recorder = ProgressRecorder(self)
    progress_description = f'Обрабатывается {total_counter} объектов'
    progress_percent = 0

    current = 0
    results = {}
    ai_prompt = system_instruction

    for index, url in enumerate(urls):
        results[f"Тест {index + 1}"] = {
            "url": url.url,
            "page_content": url.latest_content_body
        }
        if not url.latest_content_body:
            continue
        test_data = create_test_data(content=url.latest_content_body, ai_prompt=ai_prompt)
        test_data = test_data.strip("json")
        try:
            test_data = json.loads(test_data)
            benchmark_question = test_data.get("question")
            benchmark_answer = test_data.get("answer")
            if not all((benchmark_question, benchmark_answer)):
                raise json.JSONDecodeError
            results[f"Тест {index + 1}"]["benchmark_question"] = benchmark_question
            results[f"Тест {index + 1}"]["benchmark_answer"] = benchmark_answer

            # ai_documents, ai_answer = answer_index(system_instruction, benchmark_question, vectorstore)
            ai_documents, ai_answer = answer_index(system_instruction, benchmark_question, verbose=False)
            # ai_documents_serialized = [{"metadata": doc.metadata, "content": doc.page_content, } for doc in ai_documents]
            ai_documents_serialized = [
                {"score": float(doc_score), "metadata": doc.metadata, "content": doc.page_content, } for
                doc, doc_score in ai_documents]
            results[f"Тест {index + 1}"]["ai_documents"] = ai_documents_serialized
            results[f"Тест {index + 1}"]["ai_answer"] = ai_answer
            evaluation_report = get_model_response_evaluation(
                content=url.latest_content_body,
                benchmark_question=benchmark_question,
                benchmark_answer=benchmark_answer,
                ai_answer=ai_answer,
                ai_prompt=system_instruction
            )
            try:
                evaluation_report = json.loads(evaluation_report)
                results[f"Тест {index + 1}"]["evaluation_report"] = evaluation_report
            except json.JSONDecodeError:
                results[f"Тест {index + 1}"]["error"] = (f"Ответ OpenAI c оценкой не является "
                                                         f"валидным JSON! {evaluation_report}")

        except json.JSONDecodeError:
            results[f"Тест {index + 1}"]["error"] = (f"Ответ OpenAI c тестовой парой не является "
                                                     f"валидным JSON! {test_data}")
            continue

        # Обновление прогресса по процентам
        new_percent = int(((index + 1) / total_counter) * 100)
        if new_percent > progress_percent:
            progress_percent = new_percent
            progress_recorder.set_progress(progress_percent, 100, description=progress_description)

    with open("test_report.json", mode="w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return "Обработка завершена"
