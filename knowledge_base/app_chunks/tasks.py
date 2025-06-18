import json

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.db.models import Subquery, OuterRef, Max, F, Value, Prefetch
from openai import OpenAI

from app_sources.content_models import URLContent
from app_sources.source_models import URL
from test_db_zone.create_and_request_vectordb.test_db import system_instruction, answer_index, frida_vector_db


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
    {"answer_score": "оценка Ответ тестируемого AI на соответствие Эталонному ответу от 0 до 10",
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

            ai_documents, ai_answer = answer_index(system_instruction, benchmark_question, frida_vector_db)
            ai_documents_serialized = [{"metadata": doc.metadata, "content": doc.page_content, } for doc in ai_documents]
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