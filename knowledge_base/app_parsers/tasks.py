import hashlib
import json
import logging
import queue
import time
from math import ceil
from pprint import pprint

import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Dict, Any, List, Optional

import requests
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from retrying import retry
from selenium.webdriver.support.wait import WebDriverWait

from app_parsers.models import Parser, TestParser, MainParser
from app_parsers.services.parsers.base import BaseWebParser
from app_parsers.services.parsers.core_parcer_engine import SeleniumDriver
from app_parsers.services.parsers.dispatcher import WebParserDispatcher

from django.contrib.auth import get_user_model

from app_sources.content_models import URLContent, ContentStatus
from app_sources.report_models import WebSiteUpdateReport
from app_sources.source_models import URL, SourceStatus
from utils.process_text import normalize_text, remove_emoji

User = get_user_model()

logger = logging.getLogger(__name__)

# Динамическое определение CONCURRENCY_LIMIT
try:
    CONCURRENCY_LIMIT = min(psutil.cpu_count(), 5)  # Ограничение по количеству CPU
except Exception as e:
    logger.error(f"Failed to determine CPU count: {e}")
    CONCURRENCY_LIMIT = 5

driver_pool = Queue()


# Старая версия
# @shared_task(bind=True)
# def parse_urls_task(self,
#                     urls: list,
#                     parser_cls_name: str,
#                     webdriver_options: list[str] = None):
#
#     selenium = SeleniumDriver(options=webdriver_options)
#
#     # Инициализируем парсер по имени
#     parser_cls = get_parser_class_by_name(parser_cls_name)
#     parser = parser_cls(config={})
#
#     total = len(urls)
#     total_counter = len(urls)
#     if total_counter == 0:
#         return "Обработка завершена: документы не найдены"
#
#     # Инициализация прогресса
#     progress_recorder = ProgressRecorder(self)
#     progress_description = f'Обрабатывается {total_counter} объектов'
#     progress_percent = 0
#
#     current = 0
#     results = []
#
#     def fetch_page_with_selenium(url: str) -> Dict[str, Any]:
#         """
#         Загружает страницу с использованием Selenium и извлекает html.
#
#         :param url: URL страницы для загрузки.
#         :return: Словарь с результатами загрузки или ошибкой.
#         """
#         logger.info(f"page loading started {url}")
#         driver = selenium.get_driver()
#
#         try:
#             driver.get(url)  # Загрузка страницы
#             driver.execute_script("return document.readyState")  # Ожидание полной загрузки страницы
#             html = driver.page_source  # Получение HTML-контента страницы
#
#             return {
#                 "url": url,
#                 "status": 200,
#                 "html": html,
#             }
#         except Exception as e:
#             # Обработка ошибок при загрузке страницы
#             logger.error(f"page loading failed  {url}: {e}")
#             return {
#                 "url": url,
#                 "status": None,
#                 "html": None,
#             }
#         finally:
#             driver.quit()  # Закрытие Selenium-драйвера
#
#     with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
#         futures = {executor.submit(fetch_page_with_selenium, url): url for url in urls}
#         for i, future in enumerate(as_completed(futures)):
#             current += 1
#             result = future.result()
#
#             # отправляем результат в парсер
#
#             # Обновление прогресса по процентам
#             new_percent = int((current / total_counter) * 100)
#             if new_percent > progress_percent:
#                 progress_percent = new_percent
#                 progress_recorder.set_progress(progress_percent, 100, description=progress_description)
#
#     return "Обработка завершена"


def initialize_driver_pool(size: int, selenium_options: List[str]) -> None:
    """
    Инициализация пула Selenium-драйверов.

    :param size: Количество драйверов для создания в пуле.
    :type size: int
    :param selenium_options: Опции конфигурации для Selenium-драйвера.
    :type selenium_options: Dict[str, Any]
    :return: None
    """
    for _ in range(size):
        driver = SeleniumDriver(options=selenium_options).get_driver()
        driver.set_page_load_timeout(30)
        driver_pool.put(driver)


def cleanup_driver_pool() -> None:
    """
    Очистка пула драйверов, закрытие всех активных драйверов.

    :return: None
    """
    while not driver_pool.empty():
        try:
            driver = driver_pool.get(timeout=5)
            try:
                driver.quit()
            except Exception as e:
                logger.error(f"Failed to quit driver: {e}")
        except queue.Empty:
            break


# @retry(stop_max_attempt_number=3, wait_fixed=2000)
# def fetch_page_with_selenium(url: str, driver) -> Dict[str, Any]:
#     """
#     Загружает страницу с использованием Selenium и извлекает html.
#
#     :param url: URL страницы для загрузки.
#     :param driver: Selenium-драйвер из пула.
#     :return: Словарь с результатами загрузки или ошибкой.
#     """
#     import time
#     import psutil
#
#     start_time = time.time()
#     process = psutil.Process()
#     mem_before = process.memory_info().rss / 1024 / 1024  # Память в МБ
#
#     logger.info(f"Page loading started: {url}")
#     try:
#         driver.get(url)  # Загрузка страницы
#         driver.execute_script("return document.readyState")  # Ожидание полной загрузки
#         html = driver.page_source  # Получение HTML-контента
#
#         # Очистка памяти браузера
#         driver.delete_all_cookies()
#         driver.execute_script("window.localStorage.clear();")
#
#         mem_after = process.memory_info().rss / 1024 / 1024
#         logger.info(
#             f"Processed {url}, time: {time.time() - start_time:.2f}s, memory used: {mem_after - mem_before:.2f}MB")
#
#         return {
#             "url": url,
#             "status": 200,
#             "html": html,
#         }
#     except Exception as e:
#         logger.error(f"Page loading failed {url}: {e}")
#         raise  # Повторные попытки обрабатываются декоратором @retry
#
# @retry(stop_max_attempt_number=3, wait_fixed=2000)
# def fetch_page_with_selenium(url: str, driver) -> Dict[str, Any]:
#     """
#     Загружает страницу с использованием Selenium и извлекает HTML.
#
#     :param url: URL страницы для загрузки.
#     :type url: str
#     :param driver: Экземпляр Selenium-драйвера из пула.
#     :type driver: selenium.webdriver.remote.webdriver.WebDriver
#     :return: Словарь с результатами загрузки (URL, статус, HTML) или ошибкой.
#     :rtype: Dict[str, Any]
#     :raises TimeoutException: Если загрузка страницы превышает таймаут.
#     :raises Exception: При других ошибках загрузки страницы.
#     """
#     start_time: float = time.time()
#     mem_before: float = psutil.Process().memory_info().rss / 1024 / 1024
#
#     logger.info(f"Fetching {url}")
#     try:
#         driver.get(url)
#         WebDriverWait(driver, 30).until(
#             lambda d: d.execute_script("return document.readyState") == "complete"
#         )
#         html = driver.page_source
#         driver.delete_all_cookies()
#         driver.execute_script("window.localStorage.clear();")
#         mem_after: float = psutil.Process().memory_info().rss / 1024 / 1024
#
#         logger.info(
#             f"Processed {url}, time: {time.time() - start_time:.2f}s, memory used: {mem_after - mem_before:.2f}MB"
#         )
#         return {
#             "url": url,
#             "status": 200,
#             "html": html,
#         }
#     except TimeoutException as e:
#         logger.error(f"Timeout loading {url}: {e}")
#         raise
#     except Exception as e:
#         logger.error(f"Error loading {url}: {e}")
#         raise
#     finally:
#         driver.quit()


def get_http_status(url: str) -> Optional[int]:
    """Пытается получить HTTP-статус страницы через requests.head."""

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36"
        }
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=5)
        logger.info(f"HTTP status for {url}: {response.status_code}, final URL: {response.url}")
        return response.status_code
    except Exception as e:
        logger.error(f"Failed to get HTTP status for {url}: {e}")
        return 0


@retry(stop_max_attempt_number=3, wait_fixed=10000, retry_on_exception=lambda e: isinstance(e, TimeoutException))
def fetch_page_with_selenium(url: str, driver) -> Dict[str, Any]:
    """
    Загружает страницу с помощью Selenium и возвращает HTML + метаданные.

    :param url: URL страницы
    :param driver: Selenium WebDriver
    :return: Словарь с ключами:
             - url: исходный URL
             - status: HTTP-статус (None, если не удалось определить)
             - html: HTML-документ (None, если не получен)
             - error: описание ошибки, если произошла
    """
    start_time: float = time.time()
    mem_before: float = psutil.Process().memory_info().rss / 1024 / 1024
    http_status = get_http_status(url)
    html = None
    error = None

    logger.info(f"Fetching {url}")
    try:
        driver.get(url)
        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        html = driver.page_source

    except TimeoutException as e:
        error = f"Timeout loading page: {str(e)}"
        logger.error(f"[TIMEOUT] {url}: {e}")
        raise  # Повторяем попытку только для TimeoutException
    except Exception as e:
        error = f"Error loading page: {str(e)}"
        logger.error(f"[ERROR] {url}: {e}")
    finally:
        try:
            driver.delete_all_cookies()
            driver.execute_script("window.localStorage.clear();")
        except Exception as e:
            logger.warning(f"Failed to clear cookies or localStorage for {url}: {e}")

        mem_after: float = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(
            f"Processed {url}, time: {time.time() - start_time:.2f}s, memory used: {mem_after - mem_before:.2f}MB"
        )

    return {
        "url": url,
        "response_status": http_status,
        "html": html,
        "error": error
    }


def fetch_with_context(url: str, webdriver_options: list[str] = None):
    with SeleniumDriver(options=webdriver_options).driver() as driver:
        return fetch_page_with_selenium(url, driver)


@shared_task(bind=True)
def parse_urls_task(self,
                    parser: MainParser,
                    urls: list,
                    website_update_report_pk: int,
                    clean_text: bool = False,
                    clean_emoji: bool = False,
                    webdriver_options: list[str] = None):
    """
    Асинхронная задача для парсинга списка URL с использованием Selenium.

    :param website_pk: id объекта класса WebSite в рамках которого выполняется обработка
    :param mode: режим обработки "bulk" - обновление по списку, "sync" полная синхронизация объекта WebSite
    :param website_update_report_pk:
    :param main_parse_report_pk:
    :param author_id:
    :param clean_emoji:
    :param clean_text:
    :param parser:
    :param self: Объект задачи Celery.
    :param urls: Список URL для обработки.
    :param parser_cls_name: Имя класса парсера.
    :param parser_config: Настройки тестового конфига для парсера.
    :param webdriver_options: Опции для Selenium-драйвера.
    :return: Результат выполнения задачи.
    """

    total_counter = len(urls)
    if total_counter == 0:
        return "Обработка завершена: для обработки подано 0 документов"

    progress_recorder = ProgressRecorder(self)
    progress_now, current = 0, 0
    progress_step = ceil(total_counter / 100)
    progress_description = f'Обрабатывается {total_counter} объектов'
    progress_recorder.set_progress(0, 100, description=progress_description)

    parser_dispatcher = WebParserDispatcher()
    parser_cls_name = parser.class_name
    parser_cls = parser_dispatcher.get_by_class_name(parser_cls_name)
    parser_config = parser.config
    parser = parser_cls(config=parser_config if parser_config else {})

    website_update_report = WebSiteUpdateReport.objects.select_related("storage", "author").get(pk=website_update_report_pk)

    website = website_update_report.storage
    author = website_update_report.author

    parse_result = {
        'new_urls': [],
        'updated_urls': [],
        'skipped_urls': [],
        'deleted_urls': [],
        'restored_urls': [],
        'excluded_urls': [],
        'error': None
    }

    existing_urls = URL.objects.filter(url__in=urls, site_id=website.pk)
    input_urls_set = set(urls)
    existing_by_url = {obj.url: obj for obj in existing_urls}
    existing_urls_set = set(existing_by_url.keys())

    # ТОЛЬКО ДЛЯ ПОЛНОЙ СИНХРОНИЗАЦИИ САЙТА
    # deleted_urls_set = existing_urls_set - input_urls_set
    # deleted_urls = URL.objects.filter(url__in=deleted_urls_set)
    # for obj in deleted_urls:
    #     result['deleted_urls'][obj.pk] = {
    #         'url': obj.url,
    #         'title': obj.title,
    #         'status': obj.status,
    #     }

    progress_current_counter = CONCURRENCY_LIMIT

    # Инициализация пула драйверов
    # initialize_driver_pool(CONCURRENCY_LIMIT, webdriver_options)

    with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
        futures = {}
        for url in urls:
            future = executor.submit(fetch_with_context, url, webdriver_options)
            futures[future] = url

        for i, future in enumerate(as_completed(futures)):
            current += 1
            url = futures[future]
            try:
                driver_result = future.result()
                current_url = driver_result.get('url')
                response_status = driver_result.get('response_status')
                html = driver_result.get('html')
                error = driver_result.get('error')

                if html:
                    # Отправляем результат в парсер
                    parsed_data = parser.parse_html(url=url, html=html)
                    content = parsed_data.get("content", "")
                    driver_result["parsed_data"] = parsed_data
                    # Удаляем ключ "html", чтобы уменьшить объем данных
                    driver_result.pop("html", None)  # Безопасное удаление
                    # Обработка контента, если выбраны флаги
                    if clean_text or clean_emoji:
                        if clean_text:
                            content = normalize_text(content)
                        if clean_emoji:
                            content = remove_emoji(content)
                        driver_result["parsed_data"]["content"] = content
                else:
                    content = ""
                hash_content = hashlib.sha512(content.encode('utf-8')).hexdigest()

                existing_url = existing_by_url.get(url)
                if not existing_url:
                    new_url = URL.objects.create(
                        site=website,
                        report=website_update_report,
                        url=current_url,
                        status=SourceStatus.CREATED.value if response_status == 200 else SourceStatus.ERROR.value,

                    )
                    # Создаём объект URLContent, связанный с URL

                    cleaned_content = URLContent.objects.create(
                        report=website_update_report,
                        url=new_url,
                        response_status=response_status,
                        status=ContentStatus.READY.value if response_status == 200 else ContentStatus.ERROR.value,
                        author=author,
                        hash_content=hash_content,
                        body=driver_result["parsed_data"]["content"],
                        metadata=driver_result["parsed_data"].get("metadata", {}),
                        # language=None,
                        title=driver_result["parsed_data"].get("metadata", {}).get("title"),
                        tags=driver_result["parsed_data"].get("metadata", {}).get("tags", []),
                        error_message=error if error else None
                    )
                    parse_result['new_urls'].append(new_url.pk)
                else:
                    if existing_url.status == SourceStatus.DELETED.value:
                        parse_result['restored_urls'].append(existing_url.pk)
                    elif existing_url.status == SourceStatus.EXCLUDED.value:
                        parse_result['excluded_urls'].append(existing_url.pk)
                    else:
                        latest_cleaned_content = URLContent.objects.get(url=existing_url).latest()
                        if latest_cleaned_content:
                            latest_cleaned_content_hash = latest_cleaned_content.hash
                            if latest_cleaned_content_hash == hash_content:
                                parse_result['skipped_urls'].append(existing_url.pk)
                            else:
                                # TODO задача на изменение
                                cleaned_content = URLContent.objects.create(
                                    # url=existing_url,
                                    # author=author,
                                    # hash=hash_content,
                                    # body=driver_result["parsed_data"]["content"],
                                    report=website_update_report,
                                    url=new_url,
                                    response_status=response_status,
                                    status=ContentStatus.READY.value if response_status == 200 else ContentStatus.ERROR.value,
                                    author=author,
                                    hash_content=hash_content,
                                    body=driver_result["parsed_data"]["content"],
                                    metadata=driver_result["parsed_data"].get("metadata", {}),
                                    # language=None,
                                    title=driver_result["parsed_data"].get("metadata", {}).get("title"),
                                    tags=driver_result["parsed_data"].get("metadata", {}).get("tags", []),
                                    error_message=error if error else None
                                )
                                parse_result['updated_urls'].append(existing_url.pk)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                # driver_pool.put(driver)

            # Обновление прогресса
            if current >= (progress_now + 1) * progress_step:
                progress_now += 1
                progress_recorder.set_progress(progress_now, 100, description=progress_description)

    # Сохранение базовых данных
    website_update_report.author = author
    website_update_report.content["result"] = parse_result
    website_update_report.save()
    # finally:
    #     cleanup_driver_pool()  # Закрываем все драйверы

    return "Обработка завершена"

#
# @shared_task(bind=True)
# def test_single_url(self,
#                     url: str,
#                     parser: TestParser,
#                     author_id: int,
#                     clean_text: bool = False,
#                     clean_emoji: bool = False,
#                     webdriver_options: list[str] = None) -> str:
#     """
#     Celery-задача для тестирования одного URL с использованием Selenium и парсера.
#
#     :param self: Объект задачи Celery.
#     :param url: URL для обработки.
#     :param author_id: ID автора (User).
#     :param clean_text: вызывает метод очистки текста от мусора.
#     :param clean_emoji: вызывает метод очистки текста от emoji.
#     :param parser: объект класса TestParser
#     :param webdriver_options: Опции для Selenium-драйвера.
#     :return: Словарь с результатами обработки.
#     """
#     # Инициализация прогресса
#     progress_recorder = ProgressRecorder(self)
#     progress_recorder.set_progress(0, 100, description="Начало обработки URL")
#
#     parser_cls_name = parser.class_name
#     parser_config = parser.config
#     test_parse_report = parser.testparsereport
#
#     parser_dispatcher = WebParserDispatcher()
#     parser_cls = parser_dispatcher.get_by_class_name(parser_cls_name)
#     parser = parser_cls(config=parser_config if parser_config else {})
#
#     selenium = SeleniumDriver(options=webdriver_options if webdriver_options else None)
#     driver = selenium.get_driver()
#
#     result = {
#         "url": url,
#         "status": None,
#         "html": None,
#         "parsed_data": None,
#         "error": None
#     }
#
#     try:
#         progress_recorder.set_progress(50, 100, description="Страница загружается")
#         fetch_result = fetch_page_with_selenium(url, driver)
#
#         result["status"] = fetch_result["status"]
#         result["html"] = fetch_result["html"]
#
#         if fetch_result["html"]:
#             progress_recorder.set_progress(75, 100, description="Парсинг данных")
#             try:
#                 parser_result = parser.parse_html(url=url, html=fetch_result["html"])
#                 result["parsed_data"] = parser_result
#             except Exception as e:
#                 result["error"] = f"Parsing failed: {str(e)}"
#                 logger.error(f"Parsing failed for {url}: {e}")
#         else:
#             result["error"] = "No HTML content retrieved"
#
#         progress_recorder.set_progress(100, 100, description="Обработка завершена")
#         # Сохранение результата в базе данных
#         # Сохранение базовых данных
#         author = User.objects.get(pk=author_id)
#         test_parse_report.author = author
#         test_parse_report.status = result["status"]
#         test_parse_report.html = result["html"]
#         test_parse_report.parsed_data = result["parsed_data"]
#         test_parse_report.error = result["error"]
#
#         # Обработка контента, если выбраны флаги
#         if clean_text or clean_emoji:
#             content = result["parsed_data"].get("content", "")
#             if clean_text:
#                 content = normalize_text(content)
#             if clean_emoji:
#                 content = remove_emoji(content)
#             result["parsed_data"]["content"] = content
#
#         # Финальное сохранение
#         test_parse_report.parsed_data = result["parsed_data"]
#         test_parse_report.save()
#
#     except Exception as e:
#         result["error"] = f"Failed to fetch page: {str(e)}"
#         logger.error(f"Failed to process {url}: {e}")
#         progress_recorder.set_progress(100, 100, description="Обработка завершена с ошибкой")
#         test_parse_report.error = result["error"]
#         test_parse_report.save()
#
#     finally:
#         driver.quit()  # Закрытие Selenium-драйвера
#
#     return "Обработка завершена"


@shared_task(bind=True)
def test_single_url(self,
                    url: str,
                    parser: BaseWebParser,
                    report: WebSiteUpdateReport,
                    clean_text: bool = False,
                    clean_emoji: bool = False,
                    webdriver_options: list[str] = None) -> str:
    """
    Celery-задача для тестирования одного URL с использованием Selenium и парсера.
    """
    progress_recorder = ProgressRecorder(self)
    progress_recorder.set_progress(0, 100, description="Начало обработки URL")

    # parser_cls_name = parser.class_name
    # parser_config = parser.config
    # test_parse_report = parser.testparsereport
    #
    # parser_dispatcher = WebParserDispatcher()
    # parser_cls = parser_dispatcher.get_by_class_name(parser_cls_name)
    # parser = parser_cls(config=parser_config if parser_config else {})

    result = {
        "url": url,
        "status": None,
        "html": None,
        "parsed_data": None,
        "error": None
    }

    try:
        with SeleniumDriver(options=webdriver_options).driver() as driver:
            progress_recorder.set_progress(50, 100, description="Страница загружается")

            fetch_result = fetch_page_with_selenium(url, driver)

            result["status"] = fetch_result["status"]
            result["html"] = fetch_result["html"]

            if fetch_result["html"]:
                print("Предстартровая подготовка парсинга")
                progress_recorder.set_progress(75, 100, description="Парсинг данных")
                try:
                    print("Начало парсинга")
                    parser_result = parser.parse_html(url=url, html=fetch_result["html"])
                    result["parsed_data"] = parser_result
                except Exception as e:
                    print("Ошибка парсинга")
                    result["error"] = f"Parsing failed: {str(e)}"
                    logger.error(f"Parsing failed for {url}: {e}")
            else:
                result["error"] = "No HTML content retrieved"

            progress_recorder.set_progress(100, 100, description="Обработка завершена")

    except Exception as e:
        result["error"] = f"Failed to fetch page: {str(e)}"
        logger.error(f"Failed to process {url}: {e}")
        progress_recorder.set_progress(100, 100, description="Обработка завершена с ошибкой")

    # Сохранение результатов независимо от исключений
    try:
        # author = User.objects.get(pk=author_id)
        # test_parse_report.author = author
        report.status = result["status"]
        report.html = result["html"]
        report.parsed_data = result["parsed_data"]
        report.error = result["error"]

        # Обработка контента, если выбраны флаги
        if clean_text or clean_emoji:
            content = result["parsed_data"].get("content", "")
            if clean_text:
                content = normalize_text(content)
            if clean_emoji:
                content = remove_emoji(content)
            result["parsed_data"]["content"] = content
            report.parsed_data = result["parsed_data"]

        report.save()
    except Exception as e:
        logger.error(f"Failed to save test parse report for {url}: {e}")

    return "Обработка завершена"

