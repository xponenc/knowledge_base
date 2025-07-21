import json
import logging
import os
import pickle
from datetime import datetime
from pprint import pprint

import markdown

from collections import Counter

from django.db.models import Prefetch, Q
from django.shortcuts import get_object_or_404, render, redirect
from django.http import JsonResponse
from django.views import View
from django.utils import timezone
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy

from app_chat.forms import SystemInstructionForm
from app_chat.models import ChatSession, ChatMessage
from app_core.models import KnowledgeBase
from app_embeddings.forms import ModelScoreTestForm
# from app_embeddings.services.embedding_config import system_instruction, system_instruction_metadata
from app_embeddings.services.embedding_store import get_vectorstore, load_embedding
from app_embeddings.services.retrieval_engine import answer_index, answer_index_with_metadata, get_cached_multi_chain, \
    get_cached_ensemble_chain, trigram_similarity_answer_index, reformulate_question
from app_embeddings.tasks import test_model_answer
from knowledge_base.settings import BASE_DIR

from threading import Lock

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

        chat_history = chat_session.messages.filter(is_user_deleted__isnull=True).order_by("created_at")

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

        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)
        embedding_engine = kb.engine
        embeddings_model_name = embedding_engine.model_name

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

        #
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
        #         })
        #     return render(request, self.template_name, context)

        use_metadata = request.POST.get("use_metadata") == "on"
        multi_chain = get_cached_multi_chain(kb.pk)

        result = multi_chain.invoke({"input": user_message_text, "system_prompt": kb.system_instruction})
        docs = result.get("source_documents", [])
        ai_message_text = result["result"]
        verbose = True
        if verbose:
            print("Source Documents:", [doc for doc in docs])
            print("Answer:", ai_message_text)
        # if use_metadata:
        #     system_metadata_instruction = kb.system_metadata_instruction
        #     if not system_metadata_instruction:
        #         return JsonResponse({"error": "Пустая системная инструкция (вариант с метаданными)"}, status=400)
        #     docs, ai_message_text = answer_index_with_metadata(
        #         db_index,
        #         system_metadata_instruction,
        #         user_message_text,
        #         verbose=False
        #     )
        # else:
        #     system_instruction = kb.system_instruction
        #     if not system_instruction:
        #         return JsonResponse({"error": "Пустая системная инструкция"}, status=400)
        #     docs, ai_message_text = answer_index(db_index,
        #                                          system_instruction,
        #                                          user_message_text,
        #                                          verbose=False)

        # Сохраняем ответ AI
        ai_message = ChatMessage.objects.create(
            web_session=chat_session,
            answer_for=user_message,
            is_user=False,
            text=ai_message_text,
            created_at=timezone.now()
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


class SystemChatView(View):
    """Базовый чат с AI"""
    template_name = "app_chat/ai_system_chat.html"

    def get(self, request, kb_pk, *args, **kwargs):
        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)

        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key

        chat_session, _ = ChatSession.objects.get_or_create(session_key=session_key, kb=kb)

        system_instruction_form = SystemInstructionForm(
            initial={
                "system_instruction": kb.system_instruction,
            })

        chat_history = chat_session.messages.filter(is_user_deleted__isnull=True).order_by("created_at")

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
        user_message_text = request.POST.get('message', '').strip()
        if not user_message_text:
            return JsonResponse({"error": "Empty message"}, status=400)

        kb = get_object_or_404(KnowledgeBase.objects.select_related("engine"), pk=kb_pk)
        embedding_engine = kb.engine
        embeddings_model_name = embedding_engine.model_name
        is_multichain = request.POST.get("is_multichain") == "true"
        is_ensemble = request.POST.get("is_ensemble") == "true"
        is_reformulate_question = request.POST.get("is_reformulate_question") == "true"
        history_deep = request.POST.get("history_deep", None)

        system_instruction_form = SystemInstructionForm(request.POST)
        if system_instruction_form.is_valid():
            system_instruction = system_instruction_form.cleaned_data.get("system_instruction")
        else:
            system_instruction = kb.system_instruction

        if not system_instruction:
            return JsonResponse({"error": "Пустая системная инструкция"}, status=400)

        if history_deep:
            try:
                history_deep = int(history_deep)
                if history_deep > 5:
                    history_deep = 5
            except ValueError:
                history_deep = 3
        else:
            history_deep = 3

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
            created_at=timezone.now()
        )

        if is_reformulate_question and chat_session.limited_chat_history:
            chat_history = chat_session.limited_chat_history
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
                    system_instruction = system_instruction or kb.system_instruction
                    system_instruction += f"""\n
                    Документы ниже были найдены по переформулированному запросу:
                    "{reformulated_question}"
                    
                    Однако пользователь изначально спросил:
                    "{user_message_text}"
                    
                    История диалога:
                    {chat_str}
                    
                    Ответь как можно точнее на ИСХОДНЫЙ вопрос, опираясь на документы."""


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

        if is_multichain:

            multi_chain = get_cached_multi_chain(kb.pk)

            result = multi_chain.invoke({
                "input": user_message_text,
                "system_prompt": system_instruction or kb.system_instruction})
            docs = result.get("source_documents", [])
            ai_message_text = result["result"]
            # print(result)
        elif is_ensemble:
            qa_chain = get_cached_ensemble_chain(kb.pk)
            result = qa_chain.invoke({
                "input": user_message_text,
                "system_prompt": system_instruction or kb.system_instruction,
            })
            ai_message_text = result["answer"]
            docs = result.get("context", [])
        else:
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

        docs_serialized = [
            {"metadata": doc.metadata, "content": doc.page_content, }
            for doc in docs]

        # Сохраняем ответ AI
        ai_message = ChatMessage.objects.create(
            web_session=chat_session,
            answer_for=user_message,
            is_user=False,
            text=ai_message_text,
            created_at=timezone.now()
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
                'current_docs': docs_serialized,
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
            print(score)
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
        messages = (ChatMessage.objects.select_related("web_session", "t_session").prefetch_related("answer")
                    .filter(Q(web_session__kb=kb) | Q(t_session__kb=kb), is_user=True)
                    .order_by("-created_at", "web_session", "t_session"))
        context = {
            "kb": kb,
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


class TestModelScoreView(LoginRequiredMixin, View):
    """тестирование ответов"""

    def get(self, *args, **kwargs):
        all_documents = None
        chunk_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunk.pickle")
        with open(chunk_file, 'rb') as f:
            # Загружаем (десериализуем) список документов из файла
            all_documents = pickle.load(f)
        all_urls = list(set(doc.metadata.get("url") for doc in all_documents))
        form = ModelScoreTestForm(all_urls=all_urls, initial={'urls': all_urls[:5]})
        context = {
            "form": form
        }
        return render(request=self.request, template_name="app_chunks/test_model_answer.html", context=context)

    def post(self, *args, **kwargs):
        all_documents = None
        chunk_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunk.pickle")
        with open(chunk_file, 'rb') as f:
            # Загружаем (десериализуем) список документов из файла
            all_documents = pickle.load(f)
        all_urls = list(set(doc.metadata.get("url") for doc in all_documents))
        form = ModelScoreTestForm(self.request.POST, all_urls=all_urls)
        if form.is_valid():
            test_urls = form.cleaned_data.get("urls")
            task = test_model_answer.delay(test_urls=test_urls)
            return render(self.request, "celery_task_progress.html", {
                "task_id": task.id,
                "task_name": f"Тестирование ответов модели",
                "task_object_url": reverse_lazy("chunks:test_model_report"),
                "task_object_name": "FRIDA",
                "next_step_url": reverse_lazy("chunks:test_model_report"),
            })

        context = {
            "form": form
        }
        return render(request=self.request, template_name="app_chunks/test_model_answer.html", context=context)


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

