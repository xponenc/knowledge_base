import logging
from pprint import pprint

from dateutil.parser import parse
from django.contrib.auth.mixins import UserPassesTestMixin, LoginRequiredMixin
from django.core.files.base import ContentFile
from django.http import Http404
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse_lazy
from django.utils import timezone
from django.views import View
from django.contrib import messages
from django.views.generic import DetailView, ListView, CreateView, UpdateView, DeleteView

from app_core.models import KnowledgeBase
from app_parsers.forms import TestParseForm, BulkParseForm, ParserDynamicConfigForm
from app_parsers.models import Parser, TestParser, TestParseReport, MainParser, MainParserReport
from app_parsers.services.parsers.core_parcer_engine import SeleniumDriver
from app_parsers.tasks import test_single_url
from app_parsers.services.parsers.dispatcher import WebParserDispatcher
from app_sources.forms import CloudStorageForm, ContentRecognizerForm, CleanedContentEditorForm
from app_sources.models import NetworkDocument, RawContent, CleanedContent
from app_sources.storage_models import CloudStorage, CloudStorageUpdateReport, Storage, LocalStorage, WebSite
from app_sources.tasks import process_cloud_files, download_and_create_raw_content, \
    download_and_create_raw_content_parallel
from recognizers.dispatcher import ContentRecognizerDispatcher
from utils.tasks import get_task_status

logger = logging.getLogger(__name__)


class StoragePermissionMixin(UserPassesTestMixin):
    """
    Mixin для проверки прав доступа к хранилищу:
    доступ разрешён только владельцу связанной базы знаний (KnowledgeBase) или суперпользователю.
    """

    def test_func(self):
        if self.request.user.is_superuser:
            return True
        storage = self.get_object()
        kb = getattr(storage, "knowledge_base", None)
        return kb and kb.owner == self.request.user


class DocumentPermissionMixin(UserPassesTestMixin):
    """
    Mixin для проверки прав доступа к Документу:
    доступ разрешён только владельцу связанной базы знаний (KnowledgeBase) или суперпользователю.
    """

    def test_func(self):
        if self.request.user.is_superuser:
            return True
        document = self.get_object()
        storage = getattr(document, "storage", None)
        if not storage:
            return False
        kb = getattr(storage, "knowledge_base", None)
        return kb and kb.owner == self.request.user


class CloudStorageDetailView(LoginRequiredMixin, StoragePermissionMixin, DetailView):
    """Детальный просмотр объекта модели Облачное хранилище"""
    model = CloudStorage


class CloudStorageListView(LoginRequiredMixin, ListView):
    """Списковый просмотр объектов модели Облачное хранилище"""
    model = CloudStorage

    def get_queryset(self):
        queryset = super().get_queryset()
        if self.request.user.is_superuser:
            return queryset
        queryset = queryset.filter(soft_deleted_at__isnull=True).filter(owners=self.request.user)
        return queryset


class CloudStorageCreateView(LoginRequiredMixin, StoragePermissionMixin, CreateView):
    """Создание объекта модели Облачное хранилище"""
    model = CloudStorage
    form_class = CloudStorageForm

    def get_initial(self):
        """Предзаполненные значения для удобства при тестировании"""
        return {
            "name": "Облако Академии ДПО",
            "api_type": "webdav",
            "url": "https://cloud.academydpo.org/public.php/webdav/",
            "root_path": "documents/",
            "auth_type": "token",
            "token": "rqJWt7LzPGKcyNw"
        }

    def dispatch(self, request, *args, **kwargs):
        """Сохраняем knowledge_base для дальнейшего использования"""
        self.knowledge_base = get_object_or_404(KnowledgeBase, pk=kwargs['kb_pk'])
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        """Установка связи с базой знаний перед валидацией"""
        kb_pk = self.kwargs.get("kb_pk")
        if not kb_pk:
            form.add_error(None, "Не передан ID базы знаний")
            return self.form_invalid(form)

        form.instance.kb = self.knowledge_base
        form.instance.author = self.request.user
        return super().form_valid(form)

    def get_form(self, form_class=None):
        form = super().get_form(form_class)
        kb_pk = self.kwargs.get("kb_pk")
        if kb_pk:
            form.instance.kb_id = kb_pk
        form.instance.author = self.request.user
        return form

    # def get_context_data(self, **kwargs):
    #     """Добавить ID базы знаний в контекст (если нужно для шаблона)"""
    #     context = super().get_context_data(**kwargs)
    #     context['kb_pk'] = self.kwargs.get('kb_pk')
    #     return context


class CloudStorageUpdateView(LoginRequiredMixin, UpdateView):
    pass


class CloudStorageDeleteView(LoginRequiredMixin, DeleteView):
    model = CloudStorage
    success_url = reverse_lazy("sources:cloudstorage_list")


class CloudStorageSyncView(View):
    """
    Синхронизация облачного хранилища: получение списка файлов, создание отчёта, запуск фоновой задачи.
    """

    def post(self, request, pk):
        cloud_storage = get_object_or_404(CloudStorage, pk=pk)
        synced_documents = request.POST.getlist("synced_documents")
        storage_update_report = CloudStorageUpdateReport.objects.create(storage=cloud_storage, author=self.request.user)

        try:
            cloud_storage_api = cloud_storage.get_storage()
            logger.info(f"Хранилище инициализировано: {cloud_storage.name}")
            storage_update_report.content["current_status"] = "api successfully initialized"
        except ValueError as e:
            logger.error(f"Ошибка инициализации: {e}")
            storage_update_report.content["current_status"] = "api initialization failed"
            storage_update_report.content.setdefault("errors", []).append(str(e))
            storage_update_report.save()
            return render(request, 'app_sources/sync_result.html', {
                'result': {'error': f'Ошибка инициализации: {e}'},
                'cloud_storage': cloud_storage
            })

        storage_update_report.save()

        try:
            if synced_documents:
                storage_update_report.content["sync_type"] = "custom"
                storage_files = []  # cloud_storage_api.sync_selected(synced_documents)
            else:
                storage_update_report.content["sync_type"] = "all"
                # TODO надо выводить процесс в фон
                storage_files = cloud_storage_api.list_directory(path=cloud_storage_api.root_path)
                for file in storage_files:
                    file["process_status"] = "awaiting processing"
                pprint(storage_files)

            storage_update_report.content["current_status"] = "file list retrieved"
            storage_update_report.save()
            storage_update_report.save(update_fields=["content", ])

            task = process_cloud_files.delay(
                files=storage_files,
                cloud_storage=cloud_storage,
                update_report_pk=storage_update_report.pk
            )

            storage_update_report.running_background_tasks[task.id] = "Сортировка полученных файлов"
            storage_update_report.save(update_fields=["running_background_tasks", ])

            return render(request, 'app_sources/cloudstorage_progress_report.html', {
                'task_id': task.task_id,
                'cloudstorage': cloud_storage,
                "next_step_url": storage_update_report.get_absolute_url()
            })


        except Exception as e:
            logger.exception("Ошибка синхронизации файлов")
            storage_update_report.content["current_status"] = "failed to get file list from cloud"
            storage_update_report.content.setdefault("errors", []).append(str(e))
            storage_update_report.save()
            return render(request, 'app_sources/sync_result.html', {
                'result': {'error': f"Ошибка получения файлов: {e}"},
                'cloud_storage': cloud_storage
            })


class CloudStorageUpdateReportDetailView(LoginRequiredMixin, DetailView):
    """Детальный просмотр отчёта о синхронизации облачного хранилища"""
    model = CloudStorageUpdateReport

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        report = self.object
        # Добавление контекста по исполняемым фоновым задачам Celery
        running_background_tasks = report.running_background_tasks
        task_context = []
        for task_id, task_name in running_background_tasks.items():
            task_context.append(
                {
                    "task_name": task_name,
                    "task_id": task_id,
                    "report": get_task_status(task_id)
                }
            )
        context['task_context'] = task_context
        return context


class NetworkDocumentsMassCreateView(LoginRequiredMixin, View):
    """
    Создание документов (NetworkDocument) на основе отчёта синхронизации (CloudStorageUpdateReport).
    """

    def post(self, request, pk):
        selected_ids = [i for i in request.POST.getlist("file_ids") if i.strip()]
        if not selected_ids:
            # TODO message
            return redirect(reverse_lazy("sources:cloudstorageupdatereport_detail", args=[pk]))

        update_report = get_object_or_404(CloudStorageUpdateReport, pk=pk)
        new_files = update_report.content.get("result", {}).get("new_files", [])

        new_files = [file for file_id, file in new_files.items() if file_id in selected_ids]

        # Получаем все документы, которые уже есть в базе для данного хранилища
        db_documents = NetworkDocument.objects.filter(storage=update_report.storage)
        existing_urls = set(db_documents.values_list("url", flat=True))
        new_urls = {f["url"] for f in new_files}

        # Определяем дубликаты по URL
        duplicates = existing_urls & new_urls

        bulk_container = []
        # Формируем объекты для массового создания, пропуская дубликаты
        for f in new_files:
            if f["url"] in duplicates:
                f["process_status"] = "already_exists"
                continue
            try:
                remote_updated = parse(f.get("last_modified", ''))
            except Exception:
                remote_updated = None
            bulk_container.append(NetworkDocument(
                storage=update_report.storage,
                title=f["file_name"],
                path=f["path"],
                file_id=f["file_id"],
                size=f["size"],
                url=f["url"],
                remote_updated=remote_updated,
                synchronized_at=timezone.now(),
            ))

        if bulk_container:
            # Создаём все документы одним запросом
            created_docs = NetworkDocument.objects.bulk_create(bulk_container)
            created_ids = [doc.id for doc in created_docs]

            update_report.content.setdefault("created_docs", []).extend(created_ids)

            # Индекс для прохода по созданным документам
            created_doc_index = 0

            for f in new_files:
                if f.get("process_status") == "already_exists":
                    # Дубликаты пропускаем
                    continue
                # Отмечаем как созданные
                f["process_status"] = "created"
                # Привязываем id созданного документа к файлу
                f["doc_id"] = created_docs[created_doc_index].id
                created_doc_index += 1

            update_report.content["current_status"] = "Documents successfully created, download content in progress..."
            update_report.save(update_fields=["content"])

            # Запускаем фоновую задачу для скачивания и создания raw content
            task = download_and_create_raw_content_parallel.delay(
                document_ids=created_ids,
                update_report_id=update_report.pk,
                author=request.user
            )
            update_report.running_background_tasks[task.id] = "Загрузка контента файлов с облачного хранилища"
            update_report.save(update_fields=["running_background_tasks"])

        return redirect(reverse_lazy("sources:cloudstorageupdatereport_detail", args=[pk]))


class NetworkDocumentsMassUpdateView(LoginRequiredMixin, View):
    pass


class NetworkDocumentListView(LoginRequiredMixin, ListView):
    """Списковый просмотр объектов модели Сетевой документ NetworkDocument"""
    model = NetworkDocument


class NetworkDocumentDetailView(LoginRequiredMixin, DocumentPermissionMixin, DetailView):
    """Детальный просмотр объекта модели Сетевой документ NetworkDocument (с проверкой прав доступа)"""
    model = NetworkDocument


class NetworkDocumentUpdateView(LoginRequiredMixin, DocumentPermissionMixin, UpdateView):
    """Редактирование объекта модели Сетевой документ NetworkDocument (с проверкой прав доступа)"""
    model = NetworkDocument
    fields = ["title", "tags", "output_format"]


class LocalStorageListView(LoginRequiredMixin, ListView):
    """Списковый просмотр объектов модели Локальное хранилище"""
    model = LocalStorage

    def get_queryset(self):
        queryset = super().get_queryset()
        if self.request.user.is_superuser:
            return queryset
        queryset = queryset.filter(soft_deleted_at__isnull=True).filter(owners=self.request.user)
        return queryset


class LocalStorageDetailView(LoginRequiredMixin, StoragePermissionMixin, DetailView):
    """Детальный просмотр объекта модели Локальное хранилище (с проверкой прав доступа)"""
    model = LocalStorage


class LocalStorageCreateView(LoginRequiredMixin, StoragePermissionMixin, CreateView):
    """Создание объекта модели Локальное хранилище"""
    model = Storage
    fields = "__all__"


class WebSiteDetailView(LoginRequiredMixin, StoragePermissionMixin, DetailView):
    """Детальный просмотр объекта модели Вебсайт (с проверкой прав доступа)"""
    model = WebSite


class WebSiteCreateView(LoginRequiredMixin, StoragePermissionMixin, CreateView):
    """Создание объекта модели Вебсайт"""
    model = WebSite
    fields = ['name', 'base_url', 'xml_map_url']

    def dispatch(self, request, *args, **kwargs):
        """Сохраняем knowledge_base для дальнейшего использования"""
        self.knowledge_base = get_object_or_404(KnowledgeBase, pk=kwargs['kb_pk'])
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        """Установка связи с базой знаний перед валидацией"""
        kb_pk = self.kwargs.get("kb_pk")
        print("Dfkblfwbz")
        if not kb_pk:
            form.add_error(None, "Не передан ID базы знаний")
            return self.form_invalid(form)

        form.instance.kb = self.knowledge_base
        form.instance.author = self.request.user
        return super().form_valid(form)

    def get_form(self, form_class=None):
        form = super().get_form(form_class)

        kb_pk = self.kwargs.get("kb_pk")
        if kb_pk:
            form.instance.kb_id = kb_pk
        form.instance.author = self.request.user
        return form


class WebSiteUpdateView(LoginRequiredMixin, StoragePermissionMixin, UpdateView):
    """Редактирование объекта модели Вебсайт"""
    model = WebSite
    fields = ['name', 'xml_map_url']


class WebSiteDeleteView(LoginRequiredMixin, DeleteView):
    """Удаление объекта модели Вебсайт"""
    model = WebSite

    def get_success_url(self):
        return self.object.kb.get_absolute_url()


class WebSiteParseView(LoginRequiredMixin, StoragePermissionMixin, View):

    def get(self, request, pk, *args, **kwargs):
        website = get_object_or_404(WebSite, pk=pk)
        mode = request.GET.get("mode", "test")
        parser_dispatcher = WebParserDispatcher()
        all_parsers = parser_dispatcher.discover_parsers()
        current_parser = None
        parser_config_form = None

        if mode == "bulk":
            parse_form_initial = {"urls": website.base_url}
            website_main_parser = getattr(website, "mainparser", None)
            if website_main_parser:
                current_parser = website_main_parser
                # parse_form_initial["parser"] = website_main_parser.class_name
                try:
                    parser_cls = WebParserDispatcher().get_by_class_name(website_main_parser.class_name)
                    parser_config_schema = getattr(parser_cls, "config_schema", {})
                    # parser_config_form = ParserDynamicConfigForm(
                    #     schema=parser_config_schema,
                    #     initial=website_main_parser.config
                    # )
                except ValueError:
                    logger.error(
                        f"Для WebSite {website.name} не найден BaseWebParser по "
                        f"class_name = {website_main_parser.class_name}"
                    )
            parse_form = BulkParseForm(initial=parse_form_initial)

        else:  # test mode
            test_url = request.GET.get("url")
            if not test_url:
                test_url = website.base_url
            parse_form_initial = {"url": test_url}
            try:
                website_test_parser = TestParser.objects.get(site=website, author=request.user)
                print(website_test_parser)
            except TestParser.DoesNotExist:
                website_test_parser = None

            if website_test_parser:
                parse_form_initial["parser"] = website_test_parser.class_name
                try:
                    parser_cls = parser_dispatcher.get_by_class_name(website_test_parser.class_name)
                    parser_config_schema = getattr(parser_cls, "config_schema", {})
                    parser_config_form = ParserDynamicConfigForm(
                        schema=parser_config_schema,
                        initial_config=website_test_parser.config
                    )
                except ValueError:
                    logger.error(
                        f"Для WebSite {website.name} не найден BaseWebParser по "
                        f"class_name = {website_test_parser.class_name}"
                    )
            parse_form = TestParseForm(parsers=all_parsers, initial=parse_form_initial)

        return render(request, "app_sources/website_parse_form.html", {
            "form": parse_form,
            "config_form": parser_config_form,
            "mode": mode,
            "parser": current_parser,
            "website": website,
        })

    def post(self, request, pk):
        website = get_object_or_404(WebSite, id=pk)
        mode = request.GET.get("mode", "test")

        if mode == "bulk":
            parse_form = BulkParseForm(request.POST)
        else:
            parser_dispatcher = WebParserDispatcher()
            all_parsers = parser_dispatcher.discover_parsers()
            parse_form = TestParseForm(request.POST, parsers=all_parsers)

        if not parse_form.is_valid():
            return render(request, "app_sources/website_parse_form.html", {
                "form": parse_form,
                "config_form": None,
                "mode": mode,
                "website": website
            })

        clean_emoji = parse_form.cleaned_data["clean_emoji"]
        clean_text = parse_form.cleaned_data["clean_text"]

        if mode == "bulk":
            parser_config = website.mainparser.config

        else:
            parser_cls = parse_form.cleaned_data["parser"]
            parser_config_schema = getattr(parser_cls, "config_schema", {})
            parser_config_form = ParserDynamicConfigForm(request.POST, schema=parser_config_schema)

            if not parser_config_form.is_valid():
                return render(request, "app_sources/website_parse_form.html", {
                    "form": parse_form,
                    "config_form": parser_config_form,
                    "mode": mode,
                    "website": website
                })

            parser_config = parser_config_form.cleaned_data

        if mode == "bulk":
            urls = parse_form.cleaned_data["urls"]
            # main_parser, created = MainParser.create_or_update(
            #     site=website,
            #     defaults={
            #         "class_name": f"{parser_cls.__module__}.{parser_cls.__name__}",
            #         "config": parser_config,
            #         "author": request.user,
            #     }
            # )
            #
            # main_parser_report = MainParserReport.objects.create(
            #     parser=main_parser,
            #     author=request.user,
            #     content={
            #         "urls": urls,
            #         "parser_class": main_parser.class_name,
            #         "parser_config": main_parser.config,
            #     }
            # )
            print(urls)

            return render(request, "app_sources/website_parse_form.html", {
                "form": parse_form,
                "config_form": None,
                "mode": mode,
                "website": website
            })
        else:
            url = parse_form.cleaned_data["url"]
            test_parser, created = TestParser.create_or_update(
                site=website,
                author=request.user,
                defaults={
                    "class_name": f"{parser_cls.__module__}.{parser_cls.__name__}",
                    "config": parser_config,
                },

            )
            test_parser_report, created = TestParseReport.objects.get_or_create(
                parser=test_parser,
                defaults={
                    "url": url,
                    "author": request.user,
                }
            )
            if not created:
                test_parser_report.url = url
                test_parser_report.status = None
                test_parser_report.html = ""
                test_parser_report.parsed_data = {}
                test_parser_report.error = ""
                test_parser_report.author = request.user
                test_parser_report.save()

            task = test_single_url.delay(
                url=url,
                parser=test_parser,
                author_id=request.user.pk,
                webdriver_options=None,  # если не задать, то применятся дефолтные в классе
                clean_text=clean_text,
                clean_emoji=clean_emoji,
            )

            # test_single_url(
            #     url=url,
            #     parser=test_parser,
            #     author_id=request.user.pk,
            #     webdriver_options=None,  # если не задать, то применятся дефолтные в классе
            #     clean_text=clean_text,
            #     clean_emoji=clean_emoji,
            # )
            #
            # return redirect(reverse_lazy("parsers:testparser_detail", kwargs={"pk": test_parser.pk}))

            return render(request, "celery_task_progress.html", {
                "task_id": task.id,
                "task_name": f"Тестовый парсинг страницы {url}",
                "task_object_url": reverse_lazy("sources:website_detail", kwargs={"pk": website.pk}),
                "task_object_name": website.name,
                "next_step_url": reverse_lazy("parsers:testparser_detail", kwargs={"pk": test_parser.pk}),
            })


class WebSiteTestParseReportView(LoginRequiredMixin, View):
    """Класс отчета по тестовому парсингу страницы сайта"""

    def get(self, request, pk):
        website = get_object_or_404(WebSite, id=pk)
        test_report = get_object_or_404(TestParseReport, site=website, author=request.user)
        context = {
            "website": website,
            "report": test_report,
        }
        return render(request, "app_sources/testparseresult_detail.html", context)


class RawContentRecognizeCreateView(LoginRequiredMixin, View):
    def get(self, request, pk):
        raw_content = get_object_or_404(RawContent, pk=pk)
        dispatcher = ContentRecognizerDispatcher()
        file_extension = raw_content.file_extension()
        # available = dispatcher.get_recognizers_for_extension(ext)
        # print(f"{available=}")
        #
        # recognizer_choices = [(r.name, r.label) for r in available]
        recognizers = dispatcher.get_recognizers_for_extension(file_extension)
        form = ContentRecognizerForm(recognizers=recognizers)

        return render(request, "app_sources/rawcontent_recognize.html", {
            "raw_content": raw_content,
            "form": form,
        })

    def post(self, request, pk):
        raw_content = get_object_or_404(RawContent, pk=pk)
        dispatcher = ContentRecognizerDispatcher()
        file_extension = raw_content.file_extension()
        recognizers = dispatcher.get_recognizers_for_extension(file_extension)

        form = ContentRecognizerForm(request.POST, recognizers=recognizers)

        if form.is_valid():
            recognizer_class = form.cleaned_data['recognizer']
            file_path = raw_content.file.path
            print(file_path)
            # print(recognizer_name)
            # recognizer = dispatcher.get_by_name(recognizer_name)
            try:
                recognizer = recognizer_class(file_path)
                recognizer_report = recognizer.recognize()
                recognized_text = recognizer_report.get("text", "")
                recognition_method = recognizer_report.get("method", "")
                recognition_quality_report = recognizer_report.get("quality_report", {})
                if not recognized_text.strip():
                    raise ValueError("Не удалось распознать текст.")

                # Создаём CleanedContent без файла
                cleaned_content = CleanedContent.objects.create(
                    raw_content=raw_content,
                    recognition_method=recognition_method,
                    recognition_quality=recognition_quality_report,
                    author=request.user,
                )
                raw_content_source = next(
                    ((attr, getattr(raw_content, attr)) for attr in ['url', 'network_document', 'local_document'] if
                     getattr(raw_content, attr, None)),
                    (None, None)
                )
                source_name, source_value = raw_content_source
                setattr(cleaned_content, source_name, source_value)
                cleaned_content.save()
                cleaned_content.file.save("ignored.txt", ContentFile(recognized_text.encode('utf-8')))

                messages.success(request, "Очищенный контент успешно создан.")
                return redirect("sources:cleanedcontent_detail", pk=cleaned_content.pk)
            except Exception as e:
                messages.error(request, f"Ошибка при обработке файла: {e}")
        print(form.errors)
        return render(request, "app_sources/rawcontent_recognize.html", {
            "raw_content": raw_content,
            "form": form,
        })


class RawContentDetailView(LoginRequiredMixin, DetailView):
    model = RawContent


class CleanedContentDetailView(LoginRequiredMixin, DetailView):
    model = CleanedContent

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.object.file:
            try:
                with self.object.file.open('rb') as f:  # Открываем в бинарном режиме
                    raw_content = f.read()  # Читаем как байты
                    content = raw_content.decode('utf-8')  # Декодируем в строку
                    context['file_content'] = content
            except UnicodeDecodeError:
                context['file_content'] = "Не удалось прочитать содержимое файла: неподдерживаемая кодировка."

            except Exception as e:
                context['file_content'] = f"Ошибка при чтении файла: {e}"
        return context


class CleanedContentUpdateView(LoginRequiredMixin, View):
    def get(self, request, pk):
        cleaned_content = CleanedContent.objects.get(pk=pk)
        editor_content = ""
        if cleaned_content.file:
            try:
                with cleaned_content.file.open('rb') as f:  # Открываем в бинарном режиме
                    raw_content = f.read()  # Читаем как байты
                    editor_content = raw_content.decode('utf-8')  # Декодируем в строку
            except UnicodeDecodeError:
                raise Http404("Не удалось прочитать содержимое файла: неподдерживаемая кодировка.")
            except Exception as e:
                Http404("Ошибка при чтении файла")

        form = CleanedContentEditorForm(initial={"content": editor_content})
        context = {
            "form": form,
        }
        return render(request=request, template_name="app_sources/cleanedcontent_form.html", context=context)

    def post(self, request, pk):
        cleaned_content = CleanedContent.objects.get(pk=pk)
        form = CleanedContentEditorForm(request.POST)
        if form.is_valid():
            content = form.cleaned_data.get("content")
            cleaned_content.file.save("ignored.txt", ContentFile(content.encode('utf-8')))
            return redirect(reverse_lazy("sources:cleanedcontent_detail", args=[cleaned_content.pk]))
        context = {
            "form": form,
        }
        return render(request=request, template_name="app_sources/cleanedcontent_form.html", context=context)
