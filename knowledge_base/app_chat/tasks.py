import json
import re
import time

import requests
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.contrib.auth import get_user_model
from openai import OpenAI

from app_chat.models import ChatSession, ChatMessage
from app_core.models import KnowledgeBase
from telegram_bot.bot_config import KB_AI_API_KEY
from utils.setup_logger import setup_logger

User = get_user_model()
logger = setup_logger(name=__file__, log_dir="logs/chat", log_file="test_chat.log")




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
def benchmark_test_model_answer(self,
                      kb_id: int,
                      test_data: list[dict],
                      session_id: str):
    """
    Выполняет автоматическую проверку модели на сгенерированных вопросах.

    Параметры:
    - kb_id: ID базы знаний (KnowledgeBase)
    - test_data: список объектов с ключами 'storage' и 'test_data',
      где каждый элемент test_data — dict с source и content
    - session_id: ключ сессии ChatSession, в рамках которой будет вестись диалог
    """

    logger.info(f"Запущена задача test_model_answer для KB ID {kb_id} и сессии {session_id}")

    try:
        kb = KnowledgeBase.objects.get(pk=kb_id)
        logger.info(f"База знаний найдена: {kb}")
    except KnowledgeBase.DoesNotExist:
        logger.error(f"База знаний с ID {kb_id} не найдена.")
        return "Не найдена заданная База зананий"

    system_instruction = kb.system_instruction
    llm_name = kb.llm or "gpt-4o-mini"

    chat_session, _ = ChatSession.objects.get_or_create(session_key=session_id, kb=kb)
    logger.info(f"Используется сессия: {chat_session.session_key}")

    total_counter = sum(len(item.get("test_data", [])) for item in test_data)
    logger.info(f"Поступили данные для {total_counter} тестов")
    if total_counter == 0:
        logger.warning("Нет тестовых данных для обработки.")
        return "Обработка завершена: документы не найдены"

    progress_recorder = ProgressRecorder(self)
    progress_description = f'Обрабатывается {total_counter} тестов'
    progress_percent = 0

    current = 0

    for sub_test in test_data:
        storage = sub_test.get("storage")
        sub_test_data = sub_test.get("test_data")

        logger.info(f"Начата обработка хранилища: {storage} ({len(sub_test_data)} элементов)")

        for test_item_data in sub_test_data:
            current += 1
            source = test_item_data.get("source")
            content = test_item_data.get("content")

            try:
                benchmark_data = create_test_data(content=content, ai_prompt=system_instruction)
                benchmark_data = re.sub(r"^```(?:json)?\n|```$", "", benchmark_data.strip())
                logger.info(f"Автотест: {benchmark_data}")
                benchmark_data = json.loads(benchmark_data)

                benchmark_question = benchmark_data.get("question")
                benchmark_answer = benchmark_data.get("answer")

                if not all((benchmark_question, benchmark_answer)):
                    logger.warning(f"Пропущен тест: отсутствует question/answer. Источник: {source}")
                    continue

                question = ChatMessage.objects.create(
                    web_session=chat_session,
                    is_user=True,
                    text=benchmark_question,
                )

                logger.debug(f"Сформирован вопрос: {benchmark_question}")

                start_time = time.monotonic()

                response = requests.post(
                    "http://localhost:8001/api/multi-chain/invoke",
                    json={
                        "kb_id": kb.pk,
                        "query": benchmark_question,
                        "system_prompt": system_instruction or "",
                        "model": llm_name,
                    },
                    headers={"Authorization": f"Bearer {KB_AI_API_KEY}"},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()

                end = time.monotonic()
                duration = end - start_time

                ai_message_text = result["result"]
                docs = result.get("source_documents", [])

                extended_log = {
                    "test": "Random Test",
                    "source_url": str(source.get_absolute_url()),
                    "benchmark_answer": benchmark_answer,
                    "llm": llm_name,
                    "system_prompt": system_instruction,
                    "retriever_scheme": "MultiRetrievalQAChain",
                    "source_documents": docs,
                    "processing_time": duration,
                }

                ai_message = ChatMessage.objects.create(
                    web_session=chat_session,
                    answer_for=question,
                    is_user=False,
                    text=ai_message_text,
                    extended_log=extended_log,
                )

                logger.debug(f"Получен ответ от модели, длина: {len(ai_message_text)} символов")

                evaluation_report = get_model_response_evaluation(
                    content=content,
                    benchmark_question=benchmark_question,
                    benchmark_answer=benchmark_answer,
                    ai_answer=ai_message_text,
                    ai_prompt=system_instruction
                )

                try:
                    evaluation_report = json.loads(evaluation_report)
                    answer_score = evaluation_report.get("answer_score")
                    answer_resume = evaluation_report.get("answer_resume")
                    prompt_score = evaluation_report.get("prompt_score")
                    prompt_resume = evaluation_report.get("prompt_score")

                    if answer_score:
                        try:
                            answer_score = int(answer_score)
                            if -2 <= answer_score <= 2:
                                ai_message.score = answer_score
                        except ValueError:
                            logger.warning(f"Ошибка преобразования score в int: {answer_score}")

                    if answer_resume:
                        ai_message.extended_log["ai_answer_resume"] = answer_resume
                    if prompt_score:
                        ai_message.extended_log["ai_prompt_score"] = prompt_score
                    if prompt_resume:
                        ai_message.extended_log["ai_prompt_resume"] = prompt_resume

                    ai_message.save()

                except json.JSONDecodeError:
                    logger.error("Ошибка JSON при парсинге evaluation_report")

            except requests.RequestException as e:
                logger.exception(f"Ошибка запроса к LLM API: {e}")
            except json.JSONDecodeError:
                logger.error("Ошибка JSON при генерации вопроса/ответа")
            except Exception as e:
                logger.exception(f"Непредвиденная ошибка на шаге тестирования: {e}")

            # Обновление прогресса
            new_percent = int((current / total_counter) * 100)
            if new_percent > progress_percent:
                progress_percent = new_percent
                progress_recorder.set_progress(progress_percent, 100, description=progress_description)

    logger.info("Автотестирование завершено.")
    return "Обработка завершена"



@shared_task(bind=True)
def bulk_test_model_answer(self,
                      kb_id: int,
                      questions: list[str],
                      llm_name: str,
                      retriever_scheme: str,
                      session_id: str):
    """
    Выполняет тестирование модели на списке вопросов.

    Параметры:
    - kb_id: ID базы знаний (KnowledgeBase)
    - test_data: список из строк-вопросов
    - session_id: ключ сессии ChatSession, в рамках которой будет вестись диалог
    """

    logger.info(f"Запущена задача тестирования списком вопросов для KB ID {kb_id} и сессии {session_id}")

    try:
        kb = KnowledgeBase.objects.get(pk=kb_id)
        logger.info(f"База знаний найдена: {kb.name}")
    except KnowledgeBase.DoesNotExist:
        logger.error(f"База знаний с ID {kb_id} не найдена.")
        return "Не найдена заданная База зананий"

    system_instruction = kb.system_instruction
    logger.info(f"Используется системная инструкция: {system_instruction}")
    llm_name = llm_name or kb.llm or "gpt-4o-mini"
    logger.info(f"Используется llm: {llm_name}")
    retriever_scheme = retriever_scheme or kb.retriever_scheme
    logger.info(f"Используется схема ретриверов: {retriever_scheme}")

    chat_session, _ = ChatSession.objects.get_or_create(session_key=session_id, kb=kb)
    logger.info(f"Используется сессия: {chat_session.session_key}")

    total_counter = len(questions)
    logger.info(f"Поступили данные для {total_counter} тестов")
    if total_counter == 0:
        logger.warning("Нет тестовых данных для обработки.")
        return "Обработка завершена: документы не найдены"

    progress_recorder = ProgressRecorder(self)
    progress_description = f'Обрабатывается {total_counter} тестов'
    progress_percent = 0

    current = 0

    for test_question in questions:
        start_time = time.monotonic()
        logger.info(f"Начат тест по вопросу: {test_question}")
        current += 1

        question = ChatMessage.objects.create(
            web_session=chat_session,
            is_user=True,
            text=test_question,
        )
        try:
            if retriever_scheme == "multichain":
                retriever_scheme_name = "MultiRetrievalQAChain"
                response = requests.post(
                    "http://localhost:8001/api/multi-chain/invoke",
                    json={
                        "kb_id": kb.pk,
                        "query": test_question,
                        "system_prompt": system_instruction or "",
                        "model": llm_name,
                    },
                    headers={"Authorization": f"Bearer {KB_AI_API_KEY}"},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                ai_message_text = result["result"]
                docs = result.get("source_documents", [])
            elif retriever_scheme == "ensemble":
                retriever_scheme_name = "EnsembleRetriever"
                response = requests.post(
                    "http://localhost:8001/api/ensemble-chain/invoke",
                    json={
                        "kb_id": kb.pk,
                        "query": test_question,
                        "system_prompt": system_instruction or "",
                        "model": llm_name,
                    },
                    headers={
                        "Authorization": f"Bearer {KB_AI_API_KEY}",  # тот же Bearer токен
                    },
                    timeout=60,
                )

                response.raise_for_status()
                result = response.json()

                docs = result.get("source_documents", [])
                ai_message_text = result["result"]

            else:
                logger.warning(f"Некорректная схема ретриверов: {retriever_scheme}")
                continue
            end = time.monotonic()
            duration = end - start_time



            extended_log = {
                "test": "Bulk Test",
                "llm": llm_name,
                "system_prompt": system_instruction,
                "retriever_scheme": retriever_scheme_name,
                "source_documents": docs,
                "processing_time": duration,
            }

            ai_message = ChatMessage.objects.create(
                web_session=chat_session,
                answer_for=question,
                is_user=False,
                text=ai_message_text,
                extended_log=extended_log,
            )

            logger.info(f"Получен ответ от модели, длина: {len(ai_message_text)} символов")



        except requests.RequestException as e:
            logger.exception(f"Ошибка запроса к LLM API: {e}")

        except Exception as e:
            logger.exception(f"Непредвиденная ошибка на шаге тестирования: {e}")

        # Обновление прогресса
        new_percent = int((current / total_counter) * 100)
        if new_percent > progress_percent:
            progress_percent = new_percent
            progress_recorder.set_progress(progress_percent, 100, description=progress_description)

    logger.info("Автотестирование завершено.")
    return "Обработка завершена"