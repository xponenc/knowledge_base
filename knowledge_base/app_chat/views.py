import json
import logging
import os
import pickle
import random
import re
import time
from datetime import datetime
from io import TextIOWrapper
from pprint import pprint
from urllib.parse import urlencode

import markdown

from collections import Counter

import requests
from django.db.models import Prefetch, Q, Func, F, Value, CharField, FloatField, OuterRef, Subquery
from django.shortcuts import get_object_or_404, render, redirect
from django.http import JsonResponse, Http404, HttpResponse
from django.views import View
from django.utils import timezone
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import DetailView
from langchain_community.chat_models import ChatOpenAI
from rest_framework.exceptions import PermissionDenied

from app_chat.forms import SystemChatInstructionForm, KBRandomTestForm, KnowledgeBaseBulkTestForm
from app_chat.models import ChatSession, ChatMessage
from app_core.models import KnowledgeBase
from app_embeddings.services.embedding_store import load_embedding, get_vectorstore
from app_embeddings.services.question_clustering import QuestionClusterer

from app_embeddings.services.retrieval_engine import answer_index, trigram_similarity_answer_index, reformulate_question
from app_chat.tasks import benchmark_test_model_answer, bulk_test_model_answer

from threading import Lock

from app_sources.content_models import URLContent, RawContent, ContentStatus, CleanedContent
from app_sources.source_models import URL, SourceStatus, NetworkDocument, OutputDataType
from app_sources.storage_models import WebSite, CloudStorage
from knowledge_base.settings import BASE_DIR
from telegram_bot.bot_config import KB_AI_API_KEY

logger = logging.getLogger(__file__)

_model_cache = {}
_index_cache = {}
_lock = Lock()


def get_cached_model(model_name, loader_func):
    with _lock:
        if model_name not in _model_cache:
            logger.info(f"Загружаю модель эмбеддинга {model_name}")
            _model_cache[model_name] = loader_func(model_name)
        return _model_cache[model_name]


def get_cached_index(index_path: str, model_name: str, loader_func, model_obj):
    key = (index_path, model_name)
    with _lock:
        if key not in _index_cache:
            logger.info(f"Загружаю векторную базу {model_name}")
            _index_cache[key] = loader_func(index_path, model_obj)
        return _index_cache[key]


class ChatMessageDetailView(LoginRequiredMixin, DetailView):
    model = ChatMessage

    # queryset = ChatMessage.objects.select_related(
    #         "web_session__kb",
    #         "t_session__kb",
    #         "answer"
    #     )
    queryset = (
        ChatMessage.objects
        .select_related(
            "web_session__kb",
            "t_session__kb",
        ).prefetch_related(
            "answer"
        )
    )

    def get_object(self, queryset=None):
        obj = super().get_object(queryset)

        # Получаем связанную сессию (web или telegram)
        session = obj.web_session or obj.t_session
        if not session:
            raise Http404("Нет сессии у сообщения")

        # Получаем KnowledgeBase
        kb = session.kb
        self.kb = kb

        # Проверка: текущий пользователь должен быть владельцем KB
        if not kb.owners.filter(id=self.request.user.id).exists():
            raise PermissionDenied("Вы не имеете доступа к этой базе знаний")

        return obj

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context["kb"] = getattr(self, "kb", None)
        return context


class ChatView(View):
    """Базовый чат с AI"""
    template_name = "app_chat/ai_chat.html"

    def get(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)

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
            'kb': kb,
            'chat_history': messages,
        }
        return render(request, self.template_name, context)

    def post(self, request, kb_pk, *args, **kwargs):

        is_multichain = False # Настройка работы чата через MultiChainQARetriever иначе через EnsembleRetriever
        start_time = time.monotonic()
        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)
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
                    "Authorization": f"Bearer {KB_AI_API_KEY}",
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
                    "Authorization": f"Bearer {KB_AI_API_KEY}",  # тот же Bearer токен
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


class ChatClusterView(LoginRequiredMixin, View):
    """Анализ кластеров чата базы знаний"""

    def get(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase, pk=kb_pk)
        print(QuestionClusterer.result_cache)
        clusters_with_tags, data_json = QuestionClusterer.result_cache.get(kb.pk, (None, None))
        if not clusters_with_tags or not data_json:
            qc = QuestionClusterer(kb_pk=kb.pk)
            clusters_with_tags = qc.cluster_questions()
            _, data_json = QuestionClusterer.result_cache[kb.pk]

        top3 = sorted(
            clusters_with_tags.items(),
            key=lambda item: len(item[1]["docs"]),
            reverse=True
        )[:3]

        # передать в шаблон clusters_with_tags и data_json
        context = {
            "clusters_with_tags": top3,
            "cluster_data": data_json,
            "kb": kb,
        }

        return render(request, "app_chat/chat_cluster.html", context)


class ChatCreateClustersView(LoginRequiredMixin, View):
    def get(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase, pk=kb_pk)

        # Берём id и текст вопросов из БД
        user_questions = ChatMessage.objects.filter(is_user=True).values_list('id', 'text')

        qc = QuestionClusterer(kb_pk=kb.pk)
        qc.add_questions(user_questions)
        clusters = qc.cluster_questions()

        return HttpResponse(f"Начата кластеризация {len(user_questions)} вопросов для базы знаний {kb.name}")

class QwenChatView(View):
    """Базовый чат с AI"""
    template_name = "app_chat/ai_chat.html"

    def get(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)

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
            'kb': kb,
            'chat_history': messages,
        }
        return render(request, self.template_name, context)

    def post(self, request, kb_pk, *args, **kwargs):

        is_multichain = False # Настройка работы чата через MultiChainQARetriever иначе через EnsembleRetriever
        start_time = time.monotonic()
        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)
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

        retriever_scheme = "qwen"

        embedding_engine = kb.engine
        embeddings_model_name = embedding_engine.model_name
        try:
            # embeddings_model = load_embedding(embeddings_model_name)
            embeddings_model = get_cached_model(
                embeddings_model_name,
                loader_func=load_embedding
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {embeddings_model_name}: {str(e)}")
            raise ValueError(f"Ошибка загрузки модели {embeddings_model_name}: {str(e)}")

        # Инициализация или загрузка FAISS индекса
        faiss_dir = os.path.join(BASE_DIR, "media", "kb", str(kb.pk), "embedding_stores",
                                 f"WebSite_id_1_embedding_store", "FRIDA_faiss_index_db")

        try:
            # db_index = get_vectorstore(
            #     path=faiss_dir,
            #     embeddings=embeddings_model
            # )
            db_index = get_cached_index(
                index_path=faiss_dir,
                model_name=embeddings_model_name,
                loader_func=get_vectorstore,
                model_obj=embeddings_model
            )
        except Exception as e:
            logger.error(f"Ошибка векторная база {embeddings_model_name}: {str(e)}")
            context = {
                'kb': kb,
                'chat_history': [],
                'message': 'Не найдена готовая векторная база, необходимо выполнить векторизацию'
            }
            # Возвращаем JSON-ответ для AJAX
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'user_message': user_message,
                    'ai_response': '<p class="text text--alarm text--bold">Не найдена готовая векторная база,'
                                   ' необходимо выполнить векторизацию</p>',
                    'current_docs': [],
                })
            return render(request, self.template_name, context)

        docs, ai_message_text = answer_index(
            db_index,
            kb.system_instruction,
            user_message_text,
            verbose=False)

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


class SystemChatView(LoginRequiredMixin, View):
    """Базовый чат с AI"""
    template_name = "app_chat/ai_system_chat.html"

    def get(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)

        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key

        chat_session, _ = ChatSession.objects.get_or_create(session_key=session_key, kb=kb)

        system_instruction_form = SystemChatInstructionForm(instance=kb)

        chat_history = (chat_session.messages.filter(is_user_deleted__isnull=True)
                        .order_by("created_at").defer("extended_log"))

        messages = []
        for message in chat_history:
            messages.append({
                "id": message.id,
                "is_user": message.is_user,
                "text": message.text if message.is_user else markdown.markdown(message.text),
                "score": message.score,
            })

        context = {
            'system_instruction_form': system_instruction_form,
            'kb': kb,
            'chat_history': messages,
        }
        return render(request, self.template_name, context)

    def post(self, request, kb_pk, *args, **kwargs):
        start_time = time.monotonic()
        user_message_text = request.POST.get('message', '').strip()
        if not user_message_text:
            return JsonResponse({"error": "Empty message"}, status=400)

        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)

        is_multichain = request.POST.get("is_multichain") == "true"
        is_ensemble = request.POST.get("is_ensemble") == "true"
        is_reformulate_question = request.POST.get("is_reformulate_question") == "true"
        history_deep = request.POST.get("history_deep", None)

        system_instruction_form = SystemChatInstructionForm(request.POST, instance=kb)
        if system_instruction_form.is_valid():
            llm_name = system_instruction_form.cleaned_data.get("llm")
            system_instruction = system_instruction_form.cleaned_data.get("system_instruction")
        else:
            llm_name = kb.llm
            system_instruction = kb.system_instruction

        if not system_instruction:
            return JsonResponse({"error": "Пустая системная инструкция"}, status=400)

        if history_deep:
            try:
                history_deep = int(history_deep)
                if history_deep > 5:
                    history_deep = 5
            except ValueError:
                history_deep = 5
        else:
            history_deep = 5

        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key

        limited_chat_history = Prefetch(
            "messages",
            queryset=(
                ChatMessage.objects
                .prefetch_related("answer")
                .filter(is_user=True).order_by("-created_at")[:history_deep]
            ),
            to_attr="limited_chat_history",
        )
        chat_session, _ = (
            ChatSession.objects.prefetch_related(limited_chat_history).get_or_create(session_key=session_key, kb=kb)
        )

        # Сохраняем сообщение пользователя
        user_message = ChatMessage.objects.create(
            web_session=chat_session,
            is_user=True,
            text=user_message_text,
        )
        reformulated_question = ""
        if is_reformulate_question and chat_session.limited_chat_history:
            chat_history = chat_session.limited_chat_history
            if chat_history:
                history = [(msg.text, getattr(msg, "answer", None).text if getattr(msg, "answer", None) else "") for msg
                           in chat_history]
                chat_str = ""
                for user_q, ai_a in history:
                    chat_str += f"Пользователь: {user_q}\nАссистент: {ai_a}\n"
                reformulated_question, user_message_was_changed = reformulate_question(
                    current_question=user_message_text,
                    chat_history=chat_str,
                    model="gpt-4.1",
                )
                if user_message_was_changed:
                    system_instruction = system_instruction or kb.system_instruction
                    system_instruction += f"""\n
                    Документы ниже были найдены по переформулированному запросу:
                    "{reformulated_question}"
                    
                    Однако пользователь ИЗНАЧАЛЬНО спросил:
                    "{user_message_text}"
                    
                    История диалога:
                    {chat_str}
                    
                    Ответь на ИЗНАЧАЛЬНЫЙ вопрос согласно данной инструкции"""
                    system_instruction += f"\nИстория диалога: {chat_str}"

        # embedding_engine = kb.engine
        # embeddings_model_name = embedding_engine.model_name
        # try:
        #     # embeddings_model = load_embedding(embeddings_model_name)
        #     embeddings_model = get_cached_model(
        #         embeddings_model_name,
        #         loader_func=load_embedding
        #     )
        # except Exception as e:
        #     logger.error(f"Ошибка загрузки модели {embeddings_model_name}: {str(e)}")
        #     raise ValueError(f"Ошибка загрузки модели {embeddings_model_name}: {str(e)}")
        #
        # # Инициализация или загрузка FAISS индекса
        # faiss_dir = os.path.join(BASE_DIR, "media", "kb", str(kb.pk), "embedding_store",
        #                          f"{embedding_engine.name}_faiss_index_db")
        #
        # try:
        #     # db_index = get_vectorstore(
        #     #     path=faiss_dir,
        #     #     embeddings=embeddings_model
        #     # )
        #     db_index = get_cached_index(
        #         index_path=faiss_dir,
        #         model_name=embeddings_model_name,
        #         loader_func=get_vectorstore,
        #         model_obj=embeddings_model
        #     )
        # except Exception as e:
        #     logger.error(f"Ошибка векторная база {embeddings_model_name}: {str(e)}")
        #     context = {
        #         'kb': kb,
        #         'chat_history': [],
        #         'message': 'Не найдена готовая векторная база, необходимо выполнить векторизацию'
        #     }
        #     # Возвращаем JSON-ответ для AJAX
        #     if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        #         return JsonResponse({
        #             'user_message': user_message,
        #             'ai_response': '<p class="text text--alarm text--bold">Не найдена готовая векторная база, необходимо выполнить векторизацию</p>',
        #             'current_docs': [],
        #         })
        #     return render(request, self.template_name, context)
        #
        #
        # if use_metadata:
        #     docs, ai_message_text = answer_index_with_metadata(
        #         db_index,
        #         system_metadata_instruction,
        #         user_message_text,
        #         verbose=False
        #     )
        # else:
        #     docs, ai_message_text = answer_index(
        #         db_index,
        #         system_instruction,
        #         user_message_text,
        #         verbose=False)
        # docs_serialized = [
        #     {"score": float(doc_score), "metadata": doc.metadata, "content": doc.page_content, }
        #     for doc, doc_score in docs]
        llm = ChatOpenAI(model=llm_name, temperature=0)

        if is_multichain:
            retriever_scheme = "MultiRetrievalQAChain"
            # multi_chain = get_cached_multi_chain(kb.pk)
            # multi_chain = build_multi_chain(kb.pk, llm)
            # result = multi_chain.invoke({
            #     "input": user_message_text,
            #     "system_prompt": system_instruction or kb.system_instruction})
            response = requests.post(
                "http://localhost:8001/api/multi-chain/invoke",
                json={
                    # "kb_id": kb.pk,
                    "query": reformulated_question or user_message_text,
                    "system_prompt": system_instruction or kb.system_instruction,
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
            # print(result)
            # docs = [
            #     {"metadata": doc.metadata, "content": doc.page_content, }
            #     for doc in docs]
        elif is_ensemble:
            retriever_scheme = "EnsembleRetriever"

            # qa_chain = get_cached_ensemble_chain(kb.pk)
            # result = qa_chain.invoke({
            #     "input": user_message_text,
            #     "system_prompt": system_instruction or kb.system_instruction,
            # })
            # ensemble_chain = build_ensemble_chain(kb.pk, llm)
            #
            # result = ensemble_chain.invoke({
            #     "input": user_message_text,
            #     "system_prompt": system_instruction or kb.system_instruction
            # })

            response = requests.post(
                "http://localhost:8001/api/ensemble-chain/invoke",
                json={
                    "kb_id": kb.pk,
                    "query": reformulated_question or user_message_text,
                    "system_prompt": system_instruction or kb.system_instruction,
                    "model": llm_name,
                },
                headers={
                    "Authorization": f"Bearer {KB_AI_API_KEY}",  # тот же Bearer токен
                },
                timeout=60,
            )

            response.raise_for_status()
            result = response.json()

            # ai_message_text = result["answer"]
            # docs = result.get("context", [])

            docs = result.get("source_documents", [])
            ai_message_text = result["result"]

            # docs = [
            #     {"metadata": doc.metadata, "content": doc.page_content, }
            #     for doc in docs]
        else:
            retriever_scheme = "PostgreSQL TrigramSimilarity"

            result = trigram_similarity_answer_index(kb.pk,
                                                     system=system_instruction or kb.system_instruction,
                                                     query=user_message_text,
                                                     verbose=False)
            docs, ai_message_text = result

        verbose = False
        if verbose:
            print("Source Documents:")
            for doc in docs:
                pprint(doc)
            print("Answer:", ai_message_text)

        # docs_serialized = [
        #     {"metadata": doc.metadata, "content": doc.page_content, }
        #     for doc in docs]
        end = time.monotonic()
        duration = end - start_time
        # Сохраняем ответ AI
        extended_log = {
            "llm": llm_name,
            "system_prompt": system_instruction,
            "retriever_scheme": retriever_scheme,
            # "source_documents": [
            #     {
            #         "metadata": doc.metadata,
            #         "page_content": doc.page_content,
            #     }
            #     for doc in result.get("source_documents", [])
            # ]
            "source_documents": docs,
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
                    "id": 123,
                    "score": None,
                    "request_url": reverse_lazy("chat:message_score", kwargs={"message_pk": ai_message.pk}),
                    "text": ai_message_text,
                },
                'current_docs': docs,
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

        return JsonResponse({"success": True,
                             "score": score,
                             })


class ClearChatView(LoginRequiredMixin, View):
    def post(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase, pk=kb_pk)

        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key

        chat_session, _ = ChatSession.objects.get_or_create(session_key=session_key, kb=kb)

        chat_session.messages.update(is_user_deleted__isnull=datetime.now())

        return redirect(reverse_lazy('chat:chat', kwargs={"kb_pk": kb_pk}))


class ChatReportView(LoginRequiredMixin, View):
    """Просмотровый отчет по диалогам с моделью"""

    def get(self, request, kb_pk):
        kb = get_object_or_404(KnowledgeBase, pk=kb_pk)
        if not kb.is_owner_or_superuser(request.user):
            raise 404

        session_type = request.GET.get("type")
        session_id = request.GET.get("session_id")

        answer_prefetch = Prefetch(
            "answer",
            queryset=ChatMessage.objects.annotate(
                llm=Func(
                    F("extended_log"),
                    Value("llm"),
                    function="jsonb_extract_path_text",
                    output_field=CharField(),
                ),
                retriever_scheme=Func(
                    F("extended_log"),
                    Value("retriever_scheme"),
                    function="jsonb_extract_path_text",
                    output_field=CharField(),
                ),
                processing_time=Func(
                    F("extended_log"),
                    Value("processing_time"),
                    function="jsonb_extract_path_text",
                    output_field=FloatField(),
                ),
            ).defer("extended_log"),
        )

        messages = (
            ChatMessage.objects
            .select_related("web_session", "t_session")
            .prefetch_related(answer_prefetch)
            .filter(Q(web_session__kb=kb) | Q(t_session__kb=kb), is_user=True)
            .order_by("-created_at", "web_session", "t_session")
            .defer("extended_log")
        )
        filter_heading = None
        if session_type and session_type in ("web_session", "t_session") and session_id:
            if session_type == "web_session":
                messages = messages.filter(web_session__session_key=session_id)
                filter_heading = f"Отфильтровано по web сессии {session_id}"
            elif session_type == "t_session":
                messages = messages.filter(t_session__telegram_id=session_id)
                filter_heading = f"Отфильтровано по telegram сессии {session_id}"

        context = {
            "kb": kb,
            "filter_heading": filter_heading,
            "chat_messages": messages,
        }
        return render(request=request,
                      template_name="app_chat/chat_report.html",
                      context=context)


class CurrentTestChunksView(LoginRequiredMixin, View):
    def get(self, *args, **kwargs):
        all_documents = None
        chunk_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunk.pickle")
        with open(chunk_file, 'rb') as f:
            # Загружаем (десериализуем) список документов из файла
            all_documents = pickle.load(f)
        context = {"all_documents": all_documents, }
        return render(request=self.request, template_name="app_chunks/current_documents.html", context=context)


class KBRandomTestView(LoginRequiredMixin, View):
    """Тестирование случайных вопрос-ответ для заданной базы знаний"""

    def get(self, request, kb_pk, *args, **kwargs):
        kb_query = (KnowledgeBase.objects
                    .prefetch_related("website_set", "cloudstorage_set", "localstorage_set", "urlbatch_set"))
        kb = get_object_or_404(kb_query, pk=kb_pk)

        test_form = KBRandomTestForm(kb=kb)
        return render(request, "app_chat/kb_random_test_form.html", {
            "form": test_form,
            "kb": kb,
        })

    def post(self, request, kb_pk, *args, **kwargs):
        kb_query = (KnowledgeBase.objects
                    .prefetch_related("website_set", "cloudstorage_set", "localstorage_set", "urlbatch_set"))
        kb = get_object_or_404(kb_query, pk=kb_pk)
        test_form = KBRandomTestForm(request.POST, kb=kb)

        if not test_form.is_valid():
            return render(request, "app_chat/kb_random_test_form.html", {
                "form": test_form,
                "kb": kb,
            })

        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key

        selected_storages = []  # сюда соберем активные источники с количеством

        for model_name, related_name, group_label in [
            ("website", "website_set", "Сайты"),
            ("cloudstorage", "cloudstorage_set", "Облачные хранилища"),
            ("localstorage", "localstorage_set", "Локальные хранилища"),
            ("urlbatch", "urlbatch_set", "Пакеты ссылок"),
        ]:
            storages = getattr(kb, related_name).all()
            for storage in storages:
                checkbox_name = f"use_{model_name}_{storage.pk}"
                count_name = f"count_{model_name}_{storage.pk}"

                if test_form.cleaned_data.get(checkbox_name):
                    count = test_form.cleaned_data.get(count_name)
                    if count:
                        selected_storages.append({
                            "storage": storage,
                            "count": count,
                        })
        test_data = []
        for item in selected_storages:
            storage = item.get("storage")
            counter = item.get("count")
            if isinstance(storage, WebSite):
                url_ids = list(URL.objects
                               .filter(site=storage, status=SourceStatus.ACTIVE.value)
                               .values_list("id", flat=True))

                if len(url_ids) < counter:
                    raise ValueError(
                        f"Недостаточно URL для сайта {storage.name} (требуется {counter}, найдено {len(url_ids)})")

                tested_url_ids = random.sample(url_ids, counter)

                url_content_qs = URLContent.objects.filter(url_id=OuterRef("id")).order_by("-created_at")[:1]
                related_url_contents = Prefetch(
                    "urlcontent_set",
                    queryset=URLContent.objects.filter(id__in=Subquery(url_content_qs.values("id"))),
                    to_attr="related_urlcontents"
                )

                # Основной запрос
                tested_urls = URL.objects.filter(id__in=tested_url_ids).prefetch_related(
                    related_url_contents
                )
                # Назначаем active_urlcontent
                for url in tested_urls:
                    url.active_urlcontent = url.related_urlcontents[0] if url.related_urlcontents else None

                storage_test_data = [{
                    "source": url,
                    "content": url.active_urlcontent.body
                } for url in tested_urls if url.active_urlcontent]

                test_data.append({
                    "storage": storage,
                    "test_data": storage_test_data
                })
            elif isinstance(storage, CloudStorage):
                network_document_ids = list(NetworkDocument.objects
                                            .filter(storage=storage, status=SourceStatus.ACTIVE.value)
                                            .values_list("id", flat=True))

                if len(network_document_ids) < counter:
                    raise ValueError(
                        f"Недостаточно URL для сайта {storage.name} (требуется {counter}, найдено {len(network_document_ids)})")

                tested_nd_ids = random.sample(network_document_ids, counter)
                tested_network_documents = NetworkDocument.objects.filter(id__in=tested_nd_ids)

                storage_test_data = []
                for doc in tested_network_documents:
                    if doc.output_format == OutputDataType.file.value:
                        raw_content = RawContent.objects.filter(network_document=doc,
                                                                status=ContentStatus.ACTIVE.value).order_by(
                            "-created_at").first()
                        storage_test_data.append(
                            {
                                "source": doc,
                                "content": f"Документ: {raw_content.file.url}. Описание: {doc.description}"
                            }
                        )
                    else:
                        cleaned_content = (
                            CleanedContent.objects.filter(
                                network_document=doc,
                                status=ContentStatus.ACTIVE.value)
                            .order_by("-created_at").first())
                        if cleaned_content and cleaned_content.file:
                            try:
                                with cleaned_content.file.open("rb") as f:
                                    # Обернуть бинарный поток в текстовый с нужной кодировкой
                                    text_stream = TextIOWrapper(f, encoding="utf-8")
                                    content = text_stream.read(15000)
                            except UnicodeDecodeError:
                                continue
                            except Exception as e:
                                print(e)
                                continue
                        else:
                            continue
                        storage_test_data.append(
                            {
                                "source": doc,
                                "content": content
                            }
                        )

                test_data.append({
                    "storage": storage,
                    "test_data": storage_test_data
                })

        task = benchmark_test_model_answer.delay(
            test_data=test_data,
            kb_id=kb.id,
            session_id=f"random_test_{session_key}"
        )
        session_url = reverse_lazy("chat:chat_report", kwargs={"kb_pk": kb.id})
        query = urlencode({"type": "web_session", "session_id": f"random_test_{session_key}"})

        return render(self.request, "celery_task_progress.html", {
            "task_id": task.id,
            "task_name": f"Тестирование ответов модели",
            "task_object_url": f"{session_url}?{query}",
            "task_object_name": f" тестирования базы знаний {kb.name} случайными вопросами",
            "next_step_url": f"{session_url}?{query}",
        })


class KBBulkTestView(LoginRequiredMixin, View):
    """Тестирование заданной базы знаний вопросами по списку"""

    def get(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase, pk=kb_pk)

        test_form = KnowledgeBaseBulkTestForm(instance=kb)
        return render(request, "app_chat/kb_bulk_test_form.html", {
            "test_form": test_form,
            "kb": kb,
        })

    def post(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase, pk=kb_pk)
        test_form = KnowledgeBaseBulkTestForm(request.POST, request.FILES)

        if not test_form.is_valid():
            return render(request, "app_chat/kb_bulk_test_form.html", {
                "test_form": test_form,
                "kb": kb,
            })

        questions = test_form.get_questions()
        llm_name = test_form.cleaned_data.get("llm")
        retriever_scheme = test_form.cleaned_data.get("retriever_scheme")

        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key


        task = bulk_test_model_answer.delay(
            questions=questions,
            kb_id=kb.pk,
            retriever_scheme=retriever_scheme,
            llm_name=llm_name,
            session_id=f"bulk_test_{session_key}"
        )
        session_url = reverse_lazy("chat:chat_report", kwargs={"kb_pk": kb.id})
        query = urlencode({"type": "web_session", "session_id": f"bulk_test_{session_key}"})

        return render(self.request, "celery_task_progress.html", {
            "task_id": task.id,
            "task_name": f"Тестирование ответов модели",
            "task_object_url": f"{session_url}?{query}",
            "task_object_name": f" тестирования базы знаний {kb.name} списком вопросов",
            "next_step_url": f"{session_url}?{query}",
        })


class TestModelScoreReportView(LoginRequiredMixin, View):
    def get(self, *args, **kwargs):
        test_report = None
        all_documents = None
        chunk_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunk.pickle")
        with open(chunk_file, 'rb') as f:
            # Загружаем (десериализуем) список документов из файла
            all_documents = pickle.load(f)
        all_urls = list(doc.metadata.get("url") for doc in all_documents)
        all_urls_counts = Counter(all_urls)

        with open("test_report.json", encoding="utf-8") as f:
            test_report = json.load(f)
            for test_name, test_data in test_report.items():
                test_report[test_name]["chunks_counter"] = all_urls_counts.get(test_data.get("url"))
                test_report[test_name]["used_chunks"] = len(list(doc for doc in test_data.get("ai_documents", []) if
                                                                 doc.get("metadata", {}).get("url") == test_data.get(
                                                                     "url")))

        return render(self.request, 'app_chunks/test_model_results.html', {'tests': test_report})
