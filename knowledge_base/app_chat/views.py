from django.shortcuts import get_object_or_404, render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views import View
from django.utils import timezone

from app_chat.models import ChatSession, ChatMessage
from app_core.models import KnowledgeBase


class ChatView(View):
    template_name = "app_chat/ai_chat.html"

    def get(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)

        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key

        chat_session, _ = ChatSession.objects.get_or_create(session_key=session_key, kb=kb)

        messages = chat_session.messages.order_by("created_at").all()

        context = {
            'kb': kb,
            'chat_history': messages,
        }
        return render(request, self.template_name, context)

    def post(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)
        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key

        chat_session, _ = ChatSession.objects.get_or_create(session_key=session_key, kb=kb)

        user_message_text = request.POST.get('message', '').strip()
        if not user_message_text:
            return JsonResponse({"error": "Empty message"}, status=400)

        # Сохраняем сообщение пользователя
        user_message = ChatMessage.objects.create(
            session=chat_session,
            is_user=True,
            text=user_message_text,
            created_at=timezone.now()
        )

        # --- Здесь ваш код вызова AI и получение ответа ---
        # Например:
        # ai_response_text = call_your_ai_api(user_message_text)

        # Для примера пусть AI просто отвечает "Echo: {текст}"
        ai_response_text = f"Echo: {user_message_text}"

        # Сохраняем ответ AI
        ai_message = ChatMessage.objects.create(
            session=chat_session,
            is_user=False,
            text=ai_response_text,
            created_at=timezone.now()
        )

        # Возвращаем JSON с новым сообщением AI и ID для оценки
        return JsonResponse({
            "user_message": user_message.text,
            "ai_response": {
                "id": ai_message.id,
                "score": ai_message.score,  # изначально None
                "text": ai_message.text,
            },
        })


class MessageScoreView(View):
    """Установка оценки ответа AI"""

    def post(self, request, message_pk):
        try:
            score = int(request.POST.get("score"))
            if score not in range(-2, 3):  # -2, -1, 0, 1, 2
                return JsonResponse({"error": "Invalid score value"}, status=400)
        except (TypeError, ValueError):
            return JsonResponse({"error": "Score must be an integer"}, status=400)

        updated_count = ChatMessage.objects.filter(pk=message_pk, is_user=False).update(score=score)
        if not updated_count:
            return JsonResponse({"error": "Message not found or is a user message"}, status=404)

        return JsonResponse({"success": True, "score": score})