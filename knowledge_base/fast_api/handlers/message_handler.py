from django.db.models import Prefetch
from fastapi import Depends, APIRouter, HTTPException
from asgiref.sync import sync_to_async
from app_api.models import ApiClient
from app_chat.models import ChatSession, TelegramSession, ChatMessage
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
    logger.info(f"Processing message for telegram_id={data.telegram_id}, text='{data.text}'")
    history_deep = 3
    is_reformulate_question = True
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

    telegram_session, _ = await sync_to_async(TelegramSession.objects.prefetch_related(limited_chat_history).get_or_create)(
        telegram_id=data.telegram_id,
        defaults={"kb": client.knowledge_base}
    )

    user_message = await sync_to_async(ChatMessage.objects.create)(
        t_session=telegram_session,
        is_user=True,
        text=user_message_text,
    )

    logger.debug(f"User message saved: ID={user_message.pk}")

    if is_reformulate_question and telegram_session.limited_chat_history:
        chat_history = telegram_session.limited_chat_history
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
            )
            if user_message_was_changed:
                system_instruction = client.knowledge_base.system_instruction or ""
                system_instruction += f"""\n
                Документы ниже были найдены по переформулированному запросу:
                "{reformulated_question}"

                Однако пользователь изначально спросил:
                "{user_message_text}"

                История диалога:
                {chat_str}

                Ответь как можно точнее на ИСХОДНЫЙ вопрос, опираясь на документы."""

    try:
        chain = await sync_to_async(get_cached_multi_chain)(client.knowledge_base.pk)
        result = await sync_to_async(chain.invoke)({
            "input": user_message_text,
            "system_prompt": client.knowledge_base.system_instruction or ""
        })
        ai_text = result.get("result", "")
        logger.debug(f"Raw AI response: {ai_text}")
    except Exception as e:
        logger.error(f"AI processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chain error: {str(e)}")

    ai_message = await sync_to_async(ChatMessage.objects.create)(
        t_session=telegram_session,
        answer_for=user_message,
        is_user=False,
        text=ai_text,
    )
    logger.debug(f"AI message saved: ID={ai_message.pk}")

    return MessageOut(
        user_message_id=user_message.pk,
        ai_message_id=ai_message.pk,
        ai_text=ai_text,
        incoming_message_id=data.incoming_message_id,
    )