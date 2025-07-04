import json
from typing import List

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.db.models import Subquery, OuterRef, Max, F, Value, Prefetch
from openai import OpenAI

from app_chunks.models import Chunk, ChunkStatus
from app_chunks.splitters.base import BaseSplitter
from app_embeddings.services.embedding_config import system_instruction
from app_embeddings.services.retrieval_engine import answer_index
from app_sources.content_models import URLContent
from app_sources.source_models import URL
from utils.setup_logger import setup_logger

chunk_logger =  setup_logger(name="chunk_logger", log_dir = "logs", log_file = "chunking_debug.log")

def create_test_data(content:str, ai_prompt:str):
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
        content:str,
        benchmark_question:str,
        benchmark_answer:str,
        ai_answer:str,
        ai_prompt:str):
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
        results[f"Тест {index+1}"] = {
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
            ai_documents_serialized = [{"score": float(doc_score), "metadata": doc.metadata, "content": doc.page_content, } for
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


@shared_task(bind=True)
def bulk_chunks_create(
    self,
    content_list: List,
    report_pk: int,
    splitter,
    author_pk: int,
) -> str:
    """
    Разбивает веб-страницы на чанки с использованием заданного сплиттера и сохраняет их в базу данных.

    :param self: Celery-задача (для отображения прогресса).
    :param content_list: список объектов URLContent для обработки.
    :param report_pk: первичный ключ отчёта обновления.
    :param splitter: инициализированный объект класса-наследника BaseSplitter.
    :param author_id: идентификатор пользователя, создавшего чанки.
    :return: строка с информацией о завершении задачи.
    """
    progress_recorder = ProgressRecorder(self)
    total = len(content_list)
    progress_description_base = f"Создание чанков для {total} веб-страниц"
    progress_step = max(1, total // 100)

    progress_recorder.set_progress(0, total, description=progress_description_base)
    chunk_logger.info(f"[START] Разбиение {total} элементов. Отчёт ID={report_pk}")

    bulk_container = []
    batch_size = 900

    for i, content in enumerate(content_list):
        try:
            body = content.body
            metadata = content.metadata or {}
            url = content.url.url

            metadata["url"] = url

            documents = splitter.split(metadata=metadata, text_to_split=body)

            for doc_num, document in enumerate(documents):
                chunk = Chunk(
                    url_content=content,
                    status=ChunkStatus.READY.value,
                    report_id=report_pk,
                    metadata=document.metadata,
                    page_content=document.page_content,
                    splitter_cls=splitter.__class__.__name__,
                    splitter_config=splitter.config,
                    author_id=author_pk,
                )
                bulk_container.append(chunk)
                chunk_logger.debug(
                    f"[CHUNK CREATED] URLContent ID={content.id}, URL='{url}', Чанк #{doc_num + 1} из {len(documents)}"
                )

        except Exception as e:
            chunk_logger.exception(
                f"[ERROR] Не удалось создать чанки для URLContent ID={getattr(content, 'id', '?')}, "
                f"URL='{getattr(content.url, 'url', '?')}', index={i}. Ошибка: {e}"
            )
            continue

        if (i + 1) % progress_step == 0 or i + 1 == total:
            progress_recorder.set_progress(i + 1, total, description=progress_description_base)

        if len(bulk_container) >= batch_size:
            Chunk.objects.bulk_create(bulk_container)
            chunk_logger.info(f"[BULK SAVE] Сохранено {len(bulk_container)} чанков на итерации {i}")
            bulk_container.clear()

    if bulk_container:
        Chunk.objects.bulk_create(bulk_container)
        chunk_logger.info(f"[FINAL BULK SAVE] Сохранено {len(bulk_container)} чанков в финальной партии.")

    chunk_logger.info(f"[COMPLETE] Обработка завершена. Всего обработано: {total}")
    progress_recorder.set_progress(total, total, description="Генерация чанков завершена")

    return f"Успешно создано чанков для {total} URLContent."