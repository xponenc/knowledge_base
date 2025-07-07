import json
import logging
import os
import pickle
import markdown

from collections import Counter

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy
from django.views import View
from torch.backends.quantized import engine

from app_core.models import KnowledgeBase
from app_embeddings.forms import ModelScoreTestForm
from app_embeddings.services.embedding_config import system_instruction, system_instruction_metadata
from app_embeddings.services.embedding_store import get_vectorstore, load_embedding
from app_embeddings.services.retrieval_engine import answer_index, answer_index_with_metadata
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
            logger.error(f"Загружаю модель эмбеддинга {model_name}")
            _model_cache[model_name] = loader_func(model_name)
        return _model_cache[model_name]

def get_cached_index(index_path: str, model_name: str, loader_func, model_obj):
    key = (index_path, model_name)
    with _lock:
        if key not in _index_cache:
            logger.error(f"Загружаю векторную базу {model_name}")
            _index_cache[key] = loader_func(index_path, model_obj)
        return _index_cache[key]


class ChatView(LoginRequiredMixin, View):
    template_name = "app_embeddings/ai_chat.html"

    def get(self, request, kb_pk, *args, **kwargs):
        kb_queryset = KnowledgeBase.objects.select_related("engine")
        kb = get_object_or_404(kb_queryset, pk=kb_pk)

        # Получаем историю чата из сессии или создаем пустую
        chat_history = request.session.get('chat_history', {}).get(str(kb_pk), [])
        context = {
            'kb': kb,
            'chat_history': chat_history,
        }
        return render(request, self.template_name, context)

    def post(self, request, kb_pk, *args, **kwargs):
        kb_queryset = KnowledgeBase.objects.select_related("engine")
        kb = get_object_or_404(kb_queryset, pk=kb_pk)
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

        # Получаем историю чата из сессии
        chat_history = request.session.get('chat_history', {}).get(str(kb_pk), [])
        # Получаем сообщение пользователя
        user_message = request.POST.get('message', '').strip()

        # Инициализация или загрузка FAISS индекса
        faiss_dir = os.path.join(BASE_DIR, "media", "kb", str(kb.pk), "embedding_store",
                                 f"{embedding_engine.name}_faiss_index_db")

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
                'chat_history': chat_history,
                'message': 'Не найдена готовая векторная база, необходимо выполнить векторизацию'
            }
            # Возвращаем JSON-ответ для AJAX
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'user_message': user_message,
                    'ai_response': '<p class="text text--alarm text--bold">Не найдена готовая векторная база, необходимо выполнить векторизацию</p>',
                    'current_docs': [],
                })
            return render(request, self.template_name, context)



        use_metadata = request.POST.get("use_metadata") == "on"
        if user_message:
            if use_metadata:
                docs, ai_message = answer_index_with_metadata(db_index,
                                                              system_instruction_metadata,
                                                              user_message,
                                                              verbose=False)
            else:
                docs, ai_message = answer_index(db_index, system_instruction, user_message, verbose=False)
            docs_serialized = [
                {"score": float(doc_score), "metadata": doc.metadata, "content": doc.page_content, }
                for doc, doc_score in docs]
            ai_message = markdown.markdown(ai_message)
            ai_response = f"AI ответ: {ai_message}"

            # Добавляем сообщения в историю
            chat_history.append({'user': user_message, 'ai': ai_response})

            # Ограничиваем историю 5 последними сообщениями
            chat_history = chat_history[-5:]

            # Сохраняем обновленную историю в сессии
            request.session['chat_history'][str(kb_pk)] = chat_history
            request.session.modified = True

            # Возвращаем JSON-ответ для AJAX
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'user_message': user_message,
                    'ai_response': ai_response,
                    'current_docs': docs_serialized,
                })

        return render(request, self.template_name, {'chat_history': chat_history})


class ClearChatView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        # Очищаем историю чата в сессии
        if 'chat_history' in request.session:
            del request.session['chat_history']
            request.session.modified = True
        return redirect(reverse_lazy('chunks:ask_frida'))


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