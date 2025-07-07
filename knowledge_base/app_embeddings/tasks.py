import json
import logging
import os

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.contrib.auth import get_user_model
from django.db.models import OuterRef, Subquery
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from app_chunks.models import Chunk, ChunkStatus
from app_sources.content_models import URLContent
from app_sources.source_models import URL
from app_sources.storage_models import WebSite
from knowledge_base.settings import BASE_DIR
from .models import Embedding
from .services.embedding_config import system_instruction
from .services.embedding_store import load_embedding, get_vectorstore
from .services.retrieval_engine import answer_index

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