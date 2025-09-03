import json
import re
import time
from pprint import pprint

import markdown
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import PermissionDenied
from django.db.models import Prefetch
from django.db.models.expressions import result
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils.decorators import method_decorator
from django.utils.html import escape
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import DetailView, UpdateView, DeleteView
from langchain_core.messages import AIMessage
from sympy import content

from app_ai_assistants.configs.neuro_sales import NS_ASSISTANT_CONFIG
from app_ai_assistants.forms import AssistantTypeForm, BlockConfigForm
from app_ai_assistants.models import Assistant, BlockConnection, Block
from app_ai_assistants.services.block_model_validation import validate_block_config, parse_form_keys
from app_ai_assistants.services.chain_builder import build_assistant_chain, RuntimeConfigError
from app_ai_assistants.services.process_chain_report import process_chain_results
from app_ai_assistants.services.system_chat_utilites import build_assistant_runtime_forms
from app_ai_assistants.services.utils import is_markdown, format_links_markdown
from app_ai_assistants.services.visualization import generate_mermaid_for_assistant, \
    generate_cytoscape_data, build_assistant_structure
from app_api.models import ApiClient
from app_chat.models import ChatSession, ChatMessage
from app_core.models import KnowledgeBase
from app_core.service.response_cost import get_price, format_cost
from neuro_salesman.roles_config import NEURO_SALER
from neuro_salesman.utils import print_dict_structure


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
            from app_ai_assistants.services.assistant_builder import create_assistant_from_config

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


class BlockConfigUpdateView(LoginRequiredMixin, UpdateView):
    """Изменение конфигурации блока"""
    model = Block
    form_class = BlockConfigForm
    template_name = "app_ai_assistants/block_config_form.html"

    def get_queryset(self):
        # Чтобы не было повторных запросов при доступе к assistant
        return Block.objects.select_related("assistant__kb")

    def get_success_url(self):
        return self.object.assistant.get_absolute_url()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Здесь assistant уже подтянут через select_related
        context["kb"] = self.object.assistant.kb
        context["assistant"] = self.object.assistant
        return context


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

    def post(self, request, pk, *args, **kwargs):
        is_multichain = False
        history_deep = 10

        user_message_text = request.POST.get('message', '').strip()
        if not user_message_text:
            return JsonResponse({"error": "Empty message"}, status=400)

        start_time = time.monotonic()
        assistant = get_object_or_404(Assistant.objects.select_related("kb__engine"), pk=pk)
        kb = assistant.kb

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

        limited_chat_history = Prefetch(
            "messages",
            queryset=(
                ChatMessage.objects
                .prefetch_related("answer")
                .filter(is_user=True)
                .order_by("-created_at")[:history_deep]
            ),
            to_attr="limited_chat_history",
        )

        chat_session, created = (
            ChatSession.objects.prefetch_related(limited_chat_history).get_or_create(session_key=session_key, kb=kb)
        )

        # Сохраняем сообщение пользователя
        user_message = ChatMessage.objects.create(
            web_session=chat_session,
            is_user=True,
            text=user_message_text,
        )

        history = []
        last_message_from_client = ""
        last_message_from_manager = ""
        if not created:
            chat_history = list(chat_session.limited_chat_history)[::-1]
            if chat_history:
                for msg in chat_history:
                    history.append(f"Клиент: {msg.text}")
                    if not last_message_from_client:
                        last_message_from_client = msg.text
                        if not last_message_from_manager and hasattr(msg, "answer"):
                            last_message_from_manager = msg.answer.text
                    if hasattr(msg, "answer"):
                        history.append(f"Менеджер: {msg.answer.text}")

        # history.append(f"Клиент: {user_message_text}")

        assistant_session_data = chat_session.assistants_data.get(str(assistant.id), {})

        extractors_history = assistant_session_data.get("extractors_history", {})
        current_session_summary = assistant_session_data.get("summary_text", "")

        inputs = {
            "session_type": "web",
            "session_id": session_key,
            "search_retriever_type": "multi-chain" if is_multichain else "ensemble",
            "histories": history,
            "last message from manager": last_message_from_manager,
            "last message from client": user_message_text,
            "last client-manager qa": f"Клиент: {last_message_from_client}\nМенеджер: {last_message_from_manager}",
            "current_session_summary": current_session_summary,
            **extractors_history,
        }

        # ---- вот тут добавляем поддержку runtime-конфига ----
        runtime_configs = {}
        runtime_config_raw = request.POST.get("runtime_config")
        if runtime_config_raw:
            try:
                runtime_configs = json.loads(runtime_config_raw)  # dict: {block_id: {..config..}}
            except json.JSONDecodeError:
                return JsonResponse({"error": "Некорректный формат runtime_config"}, status=400)

        # передаём в билдера
        try:
            assistant_chain = build_assistant_chain(
                assistant=assistant,
                session_type="web",
                session_id=session_key,
                runtime_configs=runtime_configs,
                roles_config=NEURO_SALER,
                api_key=kb_api_key,
            )
        except RuntimeConfigError as e:
            return JsonResponse({"error": str(e)}, status=400)
        # ----------------------------------------------------

        results = assistant_chain.invoke(inputs)

        assistant_report = process_chain_results(results)

        ai_message = results.get("remove_greeting")
        ai_message_text = ai_message.content if ai_message else ""

        # session_summary_text = results.get("session_summary")
        # if session_summary_text:
        #     chat_session.summary_text = session_summary_text
        #     chat_session.save()

        # Сохранение текущей истории параметров беседы с AI ассистентом
        session_summary_text = results.get("session_summary")
        extractors_names = Block.objects.filter(assistant=assistant).values_list("name", flat=True)
        extractors_history = {}
        for extractor in extractors_names:
            extractor_history = results.get(f"{extractor}_history")
            if extractor_history:
                extractors_history[f"{extractor}_history"] = extractor_history

        chat_session.refresh_from_db()  # гарантированно свежие данные
        ChatSession.objects.filter(pk=chat_session.pk).update(
            assistants_data={assistant.pk: {
                "extractors_history": extractors_history,
                "summary_text": session_summary_text,
            }}
        )

        end_time = time.monotonic()
        duration = end_time - start_time

        extended_log = {
            "llm": llm_name,
            "retriever_scheme": assistant.name,
            "processing_time": duration,
            "assistant_data": assistant_report,
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
        ai_message_text = ai_message_text.replace("\\n", "<br>")
        if is_markdown(ai_message_text):
            ai_message_text = format_links_markdown(text=ai_message_text)
            ai_message_text = markdown.markdown(ai_message_text)

        print(f"{ai_message_text=}")

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

        chat_history = ChatMessage.objects.filter(
            session=chat_session,
            is_user_deleted__isnull=True
        ).order_by("created_at")

        return render(request, self.template_name, {'chat_history': chat_history})


class AssistantSystemChatView(View):
    """Системный чат с AI ассистентом"""
    template_name = "app_ai_assistants/ai_assistant_system_chat.html"

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

        assistant_structure = build_assistant_runtime_forms(assistant)

        context = {
            'assistant': assistant,
            'kb': kb,
            'chat_history': messages,
            'assistant_structure': assistant_structure,
        }
        return render(request, self.template_name, context)

    def post(self, request, pk, *args, **kwargs):
        history_deep = 6

        user_message_text = request.POST.get('message', '').strip()
        is_multichain = request.POST.get("is_multichain") == "true"
        is_ensemble = request.POST.get("is_ensemble") == "true"

        if not user_message_text:
            return JsonResponse({"error": "Empty message"}, status=400)

        start_time = time.monotonic()
        assistant = get_object_or_404(Assistant.objects.select_related("kb__engine"), pk=pk)
        kb = assistant.kb

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

        limited_chat_history = Prefetch(
            "messages",
            queryset=(
                ChatMessage.objects
                .prefetch_related("answer")
                .filter(is_user=True)
                .order_by("-created_at")[:history_deep]
            ),
            to_attr="limited_chat_history",
        )

        chat_session, created = (
            ChatSession.objects.prefetch_related(limited_chat_history).get_or_create(session_key=session_key, kb=kb)
        )

        # Сохраняем сообщение пользователя
        user_message = ChatMessage.objects.create(
            web_session=chat_session,
            is_user=True,
            text=user_message_text,
        )

        history = []
        last_message_from_client = ""
        last_message_from_manager = ""
        if not created:
            chat_history = list(chat_session.limited_chat_history)[::-1]
            if chat_history:
                for msg in chat_history:
                    history.append(f"Клиент: {msg.text}")
                    if not last_message_from_client:
                        last_message_from_client = msg.text
                        if not last_message_from_manager and hasattr(msg, "answer"):
                            last_message_from_manager = msg.answer.text
                    if hasattr(msg, "answer"):
                        history.append(f"Менеджер: {msg.answer.text}")

        # history.append(f"Клиент: {user_message_text}")
        assistant_session_data = chat_session.assistants_data.get(str(assistant.id), {})

        extractors_history = assistant_session_data.get("extractors_history", {})
        current_session_summary = assistant_session_data.get("summary_text", "")

        inputs = {
            "session_type": "web",
            "session_id": session_key,
            "search_retriever_type": "multi-chain" if is_multichain else "ensemble",
            "histories": history,
            "last message from manager": last_message_from_manager,
            "last message from client": user_message_text,
            "last client-manager qa": f"Клиент: {last_message_from_client}\nМенеджер: {last_message_from_manager}",
            "current_session_summary": current_session_summary,
            **extractors_history,
        }

        # ---- вот тут добавляем поддержку runtime-конфига ----
        runtime_configs = {}
        runtime_errors = {}
        form_errors = []

        runtime_config_raw = request.POST.get("runtime_config")

        if runtime_config_raw:
            try:
                runtime_configs = json.loads(runtime_config_raw)  # dict: {block_id: {..config..}}

            except json.JSONDecodeError:
                return JsonResponse({"error": "Некорректный формат runtime_config"}, status=400)

            # прогоняем валидацию каждого блока
            for block_id, config in runtime_configs.items():
                try:
                    block = Block.objects.get(pk=block_id)
                    block.config.update(config)

                except Block.DoesNotExist:
                    runtime_errors[block_id] = ["Блок не найден"]
                    continue

                try:
                    validate_block_config(block.block_type, block.config, block.name)

                except ValueError as e:
                    # вытаскиваем подробные ошибки Pydantic
                    form_errors.append(f"Блок {block.name}({block.block_type}) - ошибка конфигурации")
                    runtime_errors[block_id] = [str(e).replace("\n", "<br>")]

            if runtime_errors:
                return JsonResponse(
                    {"error": "Ошибки в runtime_config", "details": runtime_errors, "form_errors": form_errors},
                    status=400
                )

        # передаём в билдера
        try:
            assistant_chain = build_assistant_chain(
                assistant=assistant,
                session_type="web",
                session_id=session_key,
                runtime_configs=runtime_configs,
                roles_config=NEURO_SALER,
                api_key=kb_api_key,
            )
        except RuntimeConfigError as e:
            return JsonResponse({"error": str(e)}, status=400)
        # ----------------------------------------------------

        results = assistant_chain.invoke(inputs)
        #
        # print("results outputs ")
        # print_dict_structure(results)
        # print("\n")

        assistant_report = process_chain_results(results)

        ai_message = results.get("remove_greeting")
        ai_message_text = ai_message.content if ai_message else ""

        # Сохранение текущей истории параметров беседы с AI ассистентом
        session_summary_text = results.get("session_summary")
        extractors_names = Block.objects.filter(assistant=assistant).values_list("name", flat=True)
        extractors_history = {}
        for extractor in extractors_names:
            extractor_history = results.get(f"{extractor}_history")
            if extractor_history:
                extractors_history[f"{extractor}_history"] = extractor_history

        # assistants_data = chat_session.assistants_data
        # assistants_data[assistant.pk] = {
        #     "extractors_history": extractors_history,
        #     "summary_text": session_summary_text,
        # }

        print(f"{session_summary_text=}")
        print(f"{extractors_history=}")

        chat_session.refresh_from_db()  # гарантированно свежие данные
        ChatSession.objects.filter(pk=chat_session.pk).update(
            assistants_data={assistant.pk: {
                "extractors_history": extractors_history,
                "summary_text": session_summary_text,
            }}
        )

        assistant_report.update({"extractors_history": extractors_history})
        end_time = time.monotonic()
        duration = end_time - start_time

        extended_log = {
            "llm": llm_name,
            "retriever_scheme": "MultiChainRetriever" if is_multichain else "EnsembleRetriever",
            "processing_time": duration,
            "assistant_data": assistant_report,
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
        ai_message_text = ai_message_text.replace("\\n", "<br>")
        if is_markdown(ai_message_text):
            ai_message_text = format_links_markdown(text=ai_message_text)
            ai_message_text = markdown.markdown(ai_message_text)

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'user_message': user_message_text,
                'ai_response': {
                    "id": ai_message.pk,
                    "score": None,
                    "request_url": reverse_lazy("chat:message_score", kwargs={"message_pk": ai_message.pk}),
                    "text": ai_message_text,
                },
                "extended_log": extended_log,
            })

        chat_history = chat_session.messages.filter(is_user_deleted__isnull=True).order_by("created_at").defer(
            "extended_log")

        return render(request, self.template_name, {'chat_history': chat_history})
