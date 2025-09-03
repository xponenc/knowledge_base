import time

from django.db.models import Prefetch
from fastapi import Depends, APIRouter, HTTPException
from asgiref.sync import sync_to_async
from langchain_community.chat_models import ChatOpenAI

from app_api.models import ApiClient
from app_chat.models import ChatSession, TelegramSession, ChatMessage
from app_embeddings.services.ensemble_chain_factory import build_ensemble_chain
from app_embeddings.services.multi_chain_factory import build_multi_chain
from app_embeddings.services.retrieval_engine import get_cached_multi_chain, reformulate_question
from fast_api.auth import get_api_client
from fast_api.schemas.message_schemas import MessageIn, MessageOut
from utils.setup_logger import setup_logger

message_router = APIRouter()

logger = setup_logger(name=__file__, log_dir="logs/fast_api", log_file="fast_api.log")


@message_router.post("", response_model=MessageOut, summary="Обработка сообщения пользователя")
async def receive_message(data: MessageIn, client: ApiClient = Depends(get_api_client)):
    """
    Обрабатывает входящее сообщение пользователя, сохраняет его в базе данных и генерирует ответ AI.

    **Параметры**:
    - `data`: Данные сообщения (telegram_id, text, is_user, incoming_message_id).
    - `client`: Аутентифицированный клиент API (через заголовок Authorization).

    **Возвращает**:
    - `MessageOut`: Ответ, содержащий ID пользовательского и AI-сообщений, текст ответа AI и ID входящего сообщения.

    **Ошибки**:
    - 401: Неверный или отсутствующий токен авторизации.
    - 500: Ошибка при генерации ответа AI или сохранении сообщения.

    **Пример запроса**:
    ```json
    {
        "telegram_id": 123456789,
        "text": "Привет, как дела?",
        "is_user": true,
        "incoming_message_id": 123
    }
    ```

    **Пример ответа**:
    ```json
    {
        "user_message_id": 1,
        "ai_message_id": 2,
        "ai_text": "<p>Привет! Все отлично, а у тебя?</p>",
        "incoming_message_id": 123
    }
    ```
    """

    start_time = time.monotonic()
    logger.info(f"[telegram:{data.telegram_id}] Processing message: {data.text}")

    history_deep = 5
    is_reformulate_question = True

    logger.info(f"[telegram:{data.telegram_id}] Rephrasing and chat history depth are used: {history_deep=}")

    kb = client.knowledge_base
    model_name = kb.llm
    llm = ChatOpenAI(model=model_name, temperature=0)
    system_prompt = kb.system_instruction

    user_message_text = data.text

    limited_chat_history = Prefetch(
        "messages",
        queryset=(
            ChatMessage.objects
            .prefetch_related("answer")
            .filter(is_user=True).order_by("-created_at")[:history_deep]
        ),
        to_attr="limited_chat_history",
    )

    telegram_session, created = await sync_to_async(TelegramSession.objects.prefetch_related(limited_chat_history).get_or_create)(
        telegram_id=data.telegram_id,
        defaults={"kb": kb}
    )

    user_message = await sync_to_async(ChatMessage.objects.create)(
        t_session=telegram_session,
        is_user=True,
        text=user_message_text,
    )
    logger.info(f"[telegram:{data.telegram_id}] User message saved: ID={user_message.pk}")

    reformulated_question = ""

    if is_reformulate_question and not created:
        chat_history = list(telegram_session.limited_chat_history)[::-1]
        if chat_history:
            history = [(msg.text, getattr(msg, "answer", None).text if getattr(msg, "answer", None) else "") for msg
                       in chat_history]
            chat_str = ""
            for user_q, ai_a in history:
                chat_str += f"Пользователь: {user_q}\nАссистент: {ai_a}\n"
            # user_message_text = reformulate_question(
            #     current_question=user_message_text,
            #     chat_history=history,
            # )
            reformulated_question, user_message_was_changed = reformulate_question(
                current_question=user_message_text,
                chat_history=chat_str,
                model="gpt-4.1"
            )
            logger.info(f"[telegram:{data.telegram_id}] Reworded user message: {reformulated_question}")
            if user_message_was_changed:
                system_instruction = client.knowledge_base.system_instruction or ""
                system_instruction += f"""\n\nИстория диалога:\n{chat_str}"""
    try:
        retriever_scheme = "EnsembleRetriever"
        logger.info(f"[telegram:{data.telegram_id}] Call chain: {retriever_scheme}")

        chain = await sync_to_async(build_ensemble_chain)(
            kb_id=kb.id,
            llm=llm,
        )
        result = await sync_to_async(chain.invoke)({
            "input": reformulated_question or user_message_text,
            "system_prompt": system_prompt,
        })
        ai_text = result.get("answer", "")
        logger.info(f"[telegram:{data.telegram_id}] Call chain answer: {ai_text}")
    except Exception as e:
        logger.info(f"[telegram:{data.telegram_id}] Call chain error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chain error: {str(e)}")


    end_time = time.monotonic()
    duration = end_time - start_time
    extended_log = {
        "llm": model_name,
        "retriever_scheme": retriever_scheme,
        "processing_time": duration,
    }

    ai_message = await sync_to_async(ChatMessage.objects.create)(
        t_session=telegram_session,
        answer_for=user_message,
        is_user=False,
        text=ai_text,
        extended_log=extended_log,
    )
    logger.info(f"[telegram:{data.telegram_id}] AI message saved: ID={ai_message.pk}")
    logger.info(f"[telegram:{data.telegram_id}] Chain finished")
    return MessageOut(
        user_message_id=user_message.pk,
        ai_message_id=ai_message.pk,
        ai_text=ai_text,
        incoming_message_id=data.incoming_message_id,
    )