import json
import os
import pickle
import re
import tempfile
from collections import Counter

import tiktoken
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, JsonResponse
from django.views import View
from django.db.models import Subquery, OuterRef
from django.db.models.functions import Length
from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse_lazy

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app_chunks.forms import ModelScoreTestForm
from app_chunks.tasks import test_model_answer
from app_embeddings.services.embedding_config import system_instruction
from app_embeddings.services.retrieval_engine import answer_index
from app_sources.content_models import URLContent
from app_sources.source_models import URL



def split_markdown_text(markdown_text,
                        strip_headers=False):  # НЕ удалять заголовки под '#..' из page_content

    # Определяем заголовки, по которым будем разбивать текст
    headers_to_split_on = [("#", "Header 1"),  # Заголовок первого уровня
                           ("##", "Header 2"),  # Заголовок второго уровня
                           ("###", "Header 3")]  # Заголовок третьего уровня
    # Создаем экземпляр MarkdownHeaderTextSplitter с заданными заголовками
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on,
                                                   strip_headers=strip_headers)
    # Разбиваем текст на чанки в формат LangChain Document
    chunks = markdown_splitter.split_text(markdown_text)
    return chunks  # Возвращаем список чанков в формате LangChain


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Возвращает количество токенов в строке"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def split_text(text, max_count):
    headers_to_split_on = [
        ("#", "Header 1"),
        # ("##", "Header 2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    fragments = markdown_splitter.split_text(text)

    # Подсчет токенов для каждого фрагмента и построение графика
    fragment_token_counts = [num_tokens_from_string(fragment.page_content, "cl100k_base") for fragment in fragments]
    # plt.hist(fragment_token_counts, bins=50, alpha=0.5, label='Fragments')
    # plt.title('Distribution of Fragment Token Counts')
    # plt.xlabel('Token Count')
    # plt.ylabel('Frequency')
    # plt.show()
    # print(fragment_token_counts)
    # for fragment in fragments:
    #     if num_tokens_from_string(fragment.page_content, "cl100k_base") > max_count:
    #         print(fragment)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_count,
        chunk_overlap=50,
        length_function=lambda x: num_tokens_from_string(x, "cl100k_base")
    )

    source_chunks = [
        Document(page_content=chunk, metadata=fragment.metadata)
        for fragment in fragments
        for chunk in splitter.split_text(fragment.page_content)
    ]

    # Подсчет токенов для каждого source_chunk и построение графика
    source_chunk_token_counts = [num_tokens_from_string(chunk.page_content, "cl100k_base") for chunk in source_chunks]
    # plt.hist(source_chunk_token_counts, bins=20, alpha=0.5, label='Source Chunks')
    # plt.title('Distribution of Source Chunk Token Counts')
    # plt.xlabel('Token Count')
    # plt.ylabel('Frequency')
    # plt.show()
    # print(source_chunk_token_counts)

    return source_chunks, fragments


def deep_clean_metadata(obj):
    if isinstance(obj, dict):
        return {k: deep_clean_metadata(v) for k, v in obj.items() if deep_clean_metadata(v)}
    elif isinstance(obj, list):
        return [deep_clean_metadata(v) for v in obj if deep_clean_metadata(v)]
    return obj if obj not in (None, "", [], {}) else None


def process_document(doc: Document) -> Document:
    # Извлекаем page_content и metadata из документа
    page_content = doc.page_content
    metadata = doc.metadata.copy()  # Создаём копию metadata
    files = metadata.get('files', {})

    # Рекурсивная функция для обработки вложенных словарей и списков в files
    def process_files(data, page_content, prefix='files', parent_name=None):
        if isinstance(data, dict):
            for key, value in data.items():
                page_content = process_files(value, page_content, f"{prefix}__{key}", parent_name)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                # Если элемент списка — список, передаём первый элемент как имя
                if isinstance(item, list):
                    name = item[0] if len(item) > 0 else None
                    for j, sub_item in enumerate(item):
                        page_content = process_files(sub_item, page_content, f"{prefix}__{i}__{j}", name)
                else:
                    page_content = process_files(item, page_content, f"{prefix}__{i}", parent_name)
        elif isinstance(data, str) and 'http' in data:
            # Проверяем, является ли строка ссылкой
            # Ищем ссылку в page_content в формате Markdown или как чистую строку
            markdown_pattern = rf'!\[(.*?)\]\({re.escape(data)}\)'
            match = re.search(markdown_pattern, page_content)
            if match or data in page_content:
                # Используем parent_name как название, если оно есть, иначе название из Markdown или ссылку
                name = parent_name if parent_name else (match.group(1) if match else data)
                # Добавляем в metadata на корневой уровень
                metadata[prefix] = f"{name} {data}"
                # Заменяем Markdown-ссылку на название или чистую ссылку на имя
                if match:
                    page_content = re.sub(markdown_pattern, match.group(1), page_content)
                else:
                    page_content = page_content.replace(data, name)
        return page_content

    # Обрабатываем все элементы в files, обновляя page_content
    page_content = process_files(files, page_content)

    # Удаляет Markdown-заголовки (###, ##, #) в любом месте строки
    page_content = re.sub(r'\s*#{1,6}\s*', ' ', page_content)
    # Удаляем лишние пробелы, оставшиеся после замены Markdown-ссылок
    page_content = re.sub(r'\s+', ' ', page_content).strip()

    # Устанавливаем files в пустой список
    metadata['files'] = []
    metadata = deep_clean_metadata(metadata)

    # Создаём новый объект Document с обновлёнными данными
    return Document(page_content=page_content, metadata=metadata)


def flatten_metadata(d, parent_key='', sep='_'):
    """
    Рекурсивно разворачивает вложенный словарь/список в плоский словарь.
    Ключи соединяются через sep, списки индексируются.
    """
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_metadata(v, new_key, sep=sep).items())
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}"
            items.extend(flatten_metadata(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, d))
    return dict(items)


def save_documents_to_file(all_documents, filename):
    # Определяем путь к файлу, в котором будут сохранены документы
    file_path = os.path.join(os.pardir, filename)

    # Открываем файл для записи в бинарном режиме
    with open(file_path, 'wb') as f:
        # Сериализуем (сохраняем) список all_documents в файл
        pickle.dump(all_documents, f)
    print(f"Документы сохранены в файл: {file_path}")


def save_documents_to_response(request, documents, is_ajax=False):
    """Сохраняет документы во временный файл и возвращает его в HTTP-ответе с потоковой передачей."""
    if not documents:
        if is_ajax:
            return JsonResponse({"error": "No documents to save"}, status=400)
        return HttpResponse("No documents to save", status=400)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pickle', dir=os.path.dirname(os.path.abspath(__file__)))
    temp_file_path = temp_file.name

    try:
        # Сериализуем документы и закрываем файл
        pickle.dump(documents, temp_file)
        temp_file.close()

        if is_ajax:
            # Для AJAX: читаем содержимое файла и возвращаем как Blob с JSON
            file_content = temp_file.read()
            response = JsonResponse({
                "status": "success",
                "filename": "chunk.pickle",
                "content_type": "application/octet-stream"
            })
            response['Content-Disposition'] = 'attachment; filename="chunk.pickle"'
            response.content = file_content  # Передаём содержимое файла
            return response

        def file_iterator(path):
            try:
                with open(path, 'rb') as f:
                    while chunk := f.read(8192):
                        yield chunk
            finally:
                # Удаляем файл после завершения передачи
                if os.path.exists(path):
                    os.unlink(path)

        response = StreamingHttpResponse(
            file_iterator(temp_file_path),
            content_type='application/octet-stream'
        )
        response['Content-Disposition'] = 'attachment; filename="chunk.pickle"'
        return response

    except Exception as e:
        # Удаляем файл только в случае ошибки
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return HttpResponse(f"Error serializing documents: {str(e)}", status=500)

class ChunkCreateFromURLContentView(LoginRequiredMixin, View):
    """Создание чанков из URLContent"""

    def get(self, request, pk):
        url_content = URLContent.objects.select_related("url").get(pk=pk)
        url = url_content.url.url
        body = url_content.body
        metadata = url_content.metadata
        metadata.pop("internal_links")

        total_header = ""
        if metadata.get("title"):
            total_header += "Заголовок: " + metadata.get("title") + ". "
        if metadata.get("tags"):
            total_header += "Категории: " + ", ".join(metadata.get("tags")) + ". "

        # chunks = split_markdown_text(body)
        source_chunks, fragments = split_text(body, 600)
        for chunk in source_chunks:
            chunk.metadata.update(metadata)
            chunk.metadata["url"] = url

        source_chunks = [process_document(chunk) for chunk in source_chunks]

        for chunk in source_chunks:
            chunk.page_content = total_header + chunk.page_content
            chunk.metadata = flatten_metadata(chunk.metadata)

        # pprint(source_chunks)

        save_documents_to_file(source_chunks, "chunk.pickle")

        return HttpResponse("<br><br><br>".join(chunk.page_content for chunk in source_chunks))


class ChunkCreateFromWebSiteView(LoginRequiredMixin, View):
    """Создание чанков из URLContent"""

    def post(self, request, pk):
        #    # Получаем параметры из POST-запроса
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        status = request.POST.get('status')
        response_status = request.POST.get('response_status')

        def parse_int_param(param):
            try:
                return int(request.POST.get(param)) if request.POST.get(param) else None
            except (ValueError, TypeError):
                return None

        body_length_min = parse_int_param('min_body_length')
        body_length_max = parse_int_param('max_body_length')

        # Подзапросы для получения последнего URLContent
        latest_content = URLContent.objects.filter(
            url=OuterRef('pk')
        ).order_by('-created_at')

        urls_qs = URL.objects.filter(site_id=pk).annotate(
            latest_content_id=Subquery(latest_content.values('id')[:1]),
            latest_content_title=Subquery(latest_content.values('title')[:1]),
            latest_content_status=Subquery(latest_content.values('status')[:1]),
            latest_content_response_status=Subquery(latest_content.values('response_status')[:1]),
            latest_content_tags=Subquery(latest_content.values('tags')[:1]),
            latest_content_error_message=Subquery(latest_content.values('error_message')[:1]),
            latest_content_body_length=Subquery(
                latest_content.annotate(body_length=Length('body')).values('body_length')[:1]
            ),
        )

        # Фильтрация
        if status:
            urls_qs = urls_qs.filter(latest_content_status=status)
        if response_status:
            urls_qs = urls_qs.filter(latest_content_response_status=response_status)
        if body_length_min is not None:
            urls_qs = urls_qs.filter(latest_content_body_length__gte=body_length_min)
        if body_length_max is not None:
            urls_qs = urls_qs.filter(latest_content_body_length__lte=body_length_max)

        # Получаем только последние URLContent для отфильтрованных URL
        latest_ids = urls_qs.values_list("latest_content_id", flat=True)
        url_contents = URLContent.objects.select_related("url").filter(id__in=latest_ids)

        documents = []
        for url_content in url_contents:
            url = url_content.url.url
            # print(url)
            body = url_content.body
            metadata = url_content.metadata
            metadata.pop("internal_links")

            total_header = ""
            if metadata.get("title"):
                total_header += "Заголовок: " + metadata.get("title") + ". "
            if metadata.get("tags"):
                total_header += "Категории: " + ", ".join(metadata.get("tags")) + ". "

            # chunks = split_markdown_text(body)
            source_chunks, fragments = split_text(body, 750)
            for chunk in source_chunks:
                chunk.metadata.update(metadata)
                chunk.metadata["url"] = url

            source_chunks = [process_document(chunk) for chunk in source_chunks]

            for chunk in source_chunks:
                chunk.page_content = total_header + chunk.page_content
                chunk.metadata = flatten_metadata(chunk.metadata)

            documents.extend(source_chunks)

        # save_documents_to_file(documents, "chunk.pickle")
        response = save_documents_to_response(request, documents, is_ajax)
        return response


class TestAskFridaView(LoginRequiredMixin, View):
    template_name = "app_chunks/ai_chat.html"

    def get(self, request, *args, **kwargs):
        # Получаем историю чата из сессии или создаем пустую
        chat_history = request.session.get('chat_history', [])
        return render(request, self.template_name, {'chat_history': chat_history})

    def post(self, request):
        # Получаем историю чата из сессии
        chat_history = request.session.get('chat_history', [])

        # Получаем сообщение пользователя
        user_message = request.POST.get('message', '').strip()

        if user_message:
            docs, ai_message = answer_index(system_instruction, user_message, verbose=False)
            docs_serialized = [
                {"score": float(doc_score), "metadata": doc.metadata, "content": doc.page_content, }
                for doc, doc_score in docs]
            ai_response = f"AI ответ: {ai_message}"

            # Добавляем сообщения в историю
            chat_history.append({'user': user_message, 'ai': ai_response})

            # Ограничиваем историю 5 последними сообщениями
            chat_history = chat_history[-5:]

            # Сохраняем обновленную историю в сессии
            request.session['chat_history'] = chat_history
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
                test_report[test_name]["used_chunks"] = len(list(doc for doc in test_data.get("ai_documents", []) if doc.get("metadata", {}).get("url") == test_data.get("url")))

        return render(self.request, 'app_chunks/test_model_results.html', {'tests': test_report})