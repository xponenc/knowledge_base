import json

import markdown
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import PermissionDenied
from django.db.models.expressions import result
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import DetailView, UpdateView, DeleteView
from sympy import content

from app_ai_assistants.configs.neuro_sales import NS_ASSISTANT_CONFIG
from app_ai_assistants.forms import AssistantTypeForm
from app_ai_assistants.models import Assistant, BlockConnection
from app_ai_assistants.services.assistant_builder import create_assistant_from_config
from app_ai_assistants.services.visualization import generate_mermaid_for_assistant, \
    generate_cytoscape_data, build_assistant_structure
from app_chat.models import ChatSession
from app_core.models import KnowledgeBase
from neuro_salesman.roles_config import NEURO_SALER


class AssistantDetailView(LoginRequiredMixin, DetailView):
    model = Assistant
    context_object_name = "assistant"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        assistant = self.object
        context["kb"] = assistant.kb
        context["mermaid_code"] = generate_mermaid_for_assistant(assistant)
        context["cytoscape_data"] = json.dumps(
            generate_cytoscape_data(assistant)
        )
        assistant_structure = build_assistant_structure(assistant)
        context["assistant_structure"] = assistant_structure


        return context

class AssistantCreateView(LoginRequiredMixin, View):

    def get(self, request, kb_pk):
        knowledge_base = get_object_or_404(KnowledgeBase, pk=kb_pk)

        if not knowledge_base.owners.filter(pk=request.user.pk).exists():
            raise PermissionDenied("У вас нет прав на создание ассистента в этой базе знаний.")

        form = AssistantTypeForm()
        context = {
            "kb": knowledge_base,
            "form": form,
        }
        return render(request=request, template_name="app_ai_assistants/assistant_create.html", context=context)

    def post(self, request, kb_pk):
        knowledge_base = get_object_or_404(KnowledgeBase, pk=kb_pk)

        if not knowledge_base.owners.filter(pk=request.user.pk).exists():
            raise PermissionDenied("У вас нет прав на создание ассистента в этой базе знаний.")

        form = AssistantTypeForm(request.POST)
        context = {
            "kb": knowledge_base,
            "form": form,
        }
        if not form.is_valid():
            return render(request=request, template_name="app_ai_assistants/assistant_create.html", context=context)

        assistant_type = form.cleaned_data.get("type")
        if assistant_type == "neuro_sales":
            assistant, errors = create_assistant_from_config(
                kb=knowledge_base,
                author_id=request.user.pk,
                assistant_config=NS_ASSISTANT_CONFIG,
                roles_config=NEURO_SALER,
            )

            if assistant:
                return redirect(assistant.get_absolute_url())

            form.add_error(None, errors)

        return render(request=request, template_name="app_ai_assistants/assistant_create.html", context=context)


@method_decorator(csrf_exempt, name="dispatch")
class AssistantSaveGraphView(View):
    """
    Обновляет связи блоков на основе данных из Cytoscape.js
    """
    def post(self, request, pk):
        assistant = Assistant.objects.get(pk=pk)
        data = json.loads(request.body)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Перезаписываем связи
        BlockConnection.objects.filter(from_block__assistant=assistant).delete()
        for e in edges:
            BlockConnection.objects.create(
                from_block_id=int(e["source"][1:]),  # B123 → 123
                to_block_id=int(e["target"][1:]),
                order=e.get("order", 0)
            )

        return JsonResponse({"status": "ok"})

class AssistantUpdateView(LoginRequiredMixin, UpdateView):
    pass

class AssistantDeleteView(LoginRequiredMixin, DeleteView):
    pass


class AssistantChatView(View):
    """Базовый чат с AI ассистентом"""
    template_name = "app_ai_assistants/ai_assistant_chat.html"

    def get(self, request, pk, *args, **kwargs):
        assistant = get_object_or_404(Assistant.objects.select_related("kb__engine"), pk=pk)
        kb = assistant.kb

        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key

        chat_session, _ = ChatSession.objects.get_or_create(session_key=session_key, kb=kb)

        chat_history = chat_session.messages.filter(is_user_deleted__isnull=True).order_by("created_at").defer(
            "extended_log")

        messages = []
        for message in chat_history:
            messages.append({
                "id": message.id,
                "is_user": message.is_user,
                "text": message.text if message.is_user else markdown.markdown(message.text),
                "score": message.score,
            })

        context = {
            'assistant': assistant,
            'kb': kb,
            'chat_history': messages,
        }
        return render(request, self.template_name, context)

    def post(self, request, kb_pk, *args, **kwargs):
        is_multichain = False # Настройка работы чата через MultiChainQARetriever иначе через EnsembleRetriever
        start_time = time.monotonic()
        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)
        try:
            api_client = ApiClient.objects.get(knowledge_base=kb, name="internal api point")
        except ApiClient.DoesNotExist:
            return JsonResponse({"error": "Не задан ApiClient 'internal api point' для базы знаний "}, status=400)
        kb_api_key = api_client.token

        llm_name = kb.llm

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
            web_session=chat_session,
            is_user=True,
            text=user_message_text,
            created_at=timezone.now()
        )
        if is_multichain:

            retriever_scheme = "MultiRetrievalQAChain"
            response = requests.post(
                "http://localhost:8001/api/multi-chain/invoke",
                json={
                    "query": user_message_text,
                    "model": llm_name,
                },
                headers={
                    "Authorization": f"Bearer {kb_api_key}",
                },
                timeout=60,
            )

            response.raise_for_status()
            result = response.json()
            ai_message_text = result["result"]
        else:
            retriever_scheme = "EnsembleRetriever"

            response = requests.post(
                "http://localhost:8001/api/ensemble-chain/invoke",
                json={
                    "query": user_message_text,
                    "model": llm_name,
                },
                headers={
                    "Authorization": f"Bearer {kb_api_key}",  # тот же Bearer токен
                },
                timeout=60,
            )

            response.raise_for_status()
            result = response.json()

            ai_message_text = result["result"]

        # Сохраняем ответ AI
        end_time = time.monotonic()
        duration = end_time - start_time
        extended_log = {
            "llm": llm_name,
            "retriever_scheme": retriever_scheme,
            "processing_time": duration,
        }

        scheme = request.scheme
        host = request.get_host()
        ai_message_text = ai_message_text.replace("http://127.0.0.1:8000", f"{scheme}://{host}")

        ai_message = ChatMessage.objects.create(
            web_session=chat_session,
            answer_for=user_message,
            is_user=False,
            text=ai_message_text,
            extended_log=extended_log,
        )
        ai_message_text = markdown.markdown(ai_message_text)

        # Возвращаем JSON-ответ для AJAX
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'user_message': user_message_text,
                'ai_response': {
                    "id": ai_message.pk,
                    "score": None,
                    "request_url": reverse_lazy("chat:message_score", kwargs={"message_pk": ai_message.pk}),
                    "text": ai_message_text,
                },
            })
        chat_history = ChatMessage.objects.filter(session=chat_session,
                                                  is_user_deleted__isnull=True).order_by("created_at")
        messages = []
        for message in chat_history:
            messages.append({
                "id": message.id,
                "is_user": message.is_user,
                "text": message.text if message.is_user else markdown.markdown(message.text),
                "score": message.score,
            })

        return render(request, self.template_name, {'chat_history': chat_history})