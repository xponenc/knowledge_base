import json
import logging
from pprint import pprint

import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Dict, Any

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from selenium.webdriver.common.by import By
from retrying import retry

from app_parsers.models import Parser, TestParser
from app_parsers.services.parsers.base import BaseWebParser
from app_parsers.services.parsers.core_parcer_engine import SeleniumDriver
from app_parsers.services.parsers.dispatcher import WebParserDispatcher
from app_sources.storage_models import WebSite

from django.contrib.auth import get_user_model

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


def initialize_driver_pool(size, selenium_options):
    """Инициализация пула Selenium-драйверов."""
    for _ in range(size):
        driver = SeleniumDriver(options=selenium_options).get_driver()
        driver.set_page_load_timeout(30)  # Таймаут загрузки страницы
        driver_pool.put(driver)


def cleanup_driver_pool():
    """Очистка пула драйверов."""
    while not driver_pool.empty():
        driver = driver_pool.get()
        driver.quit()


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_page_with_selenium(url: str, driver) -> Dict[str, Any]:
    """
    Загружает страницу с использованием Selenium и извлекает html.

    :param url: URL страницы для загрузки.
    :param driver: Selenium-драйвер из пула.
    :return: Словарь с результатами загрузки или ошибкой.
    """
    import time
    import psutil

    start_time = time.time()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # Память в МБ

    logger.info(f"Page loading started: {url}")
    try:
        driver.get(url)  # Загрузка страницы
        driver.execute_script("return document.readyState")  # Ожидание полной загрузки
        html = driver.page_source  # Получение HTML-контента

        # Очистка памяти браузера
        driver.delete_all_cookies()
        driver.execute_script("window.localStorage.clear();")

        mem_after = process.memory_info().rss / 1024 / 1024
        logger.info(
            f"Processed {url}, time: {time.time() - start_time:.2f}s, memory used: {mem_after - mem_before:.2f}MB")

        return {
            "url": url,
            "status": 200,
            "html": html,
        }
    except Exception as e:
        logger.error(f"Page loading failed {url}: {e}")
        raise  # Повторные попытки обрабатываются декоратором @retry


# Драйвер не закрывается, так как он возвращается в пул

@shared_task(bind=True)
def parse_urls_task(self,
                    urls: list,
                    parser_cls_name: str,
                    parser_config: dict[str, Any] = None,
                    webdriver_options: list[str] = None):
    """
    Асинхронная задача для парсинга списка URL с использованием Selenium.

    :param self: Объект задачи Celery.
    :param urls: Список URL для обработки.
    :param parser_cls_name: Имя класса парсера.
    :param parser_config: Настройки тестового конфига для парсера.
    :param webdriver_options: Опции для Selenium-драйвера.
    :return: Результат выполнения задачи.
    """
    # selenium = SeleniumDriver(options=webdriver_options)
    parser_dispatcher = WebParserDispatcher()
    parser_cls = parser_dispatcher.get_by_class_name(parser_cls_name)
    parser = parser_cls(config=parser_config if parser_config else {})

    total_counter = len(urls)
    if total_counter == 0:
        return "Обработка завершена: документы не найдены"

    # Инициализация прогресса
    progress_recorder = ProgressRecorder(self)
    progress_description = f'Обрабатывается {total_counter} объектов'
    progress_percent = 0

    current = 0
    results = []

    # Инициализация пула драйверов
    initialize_driver_pool(CONCURRENCY_LIMIT, webdriver_options)

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
            futures = {executor.submit(fetch_page_with_selenium, url, driver_pool.get()): url for url in urls}
            for i, future in enumerate(as_completed(futures)):
                current += 1
                result = future.result()
                driver_pool.put(future._state['driver'])  # Возвращаем драйвер в пул

                # Отправляем результат в парсер
                parsed_data = parser.parse(result.get('html', None))
                results.append(parsed_data)

                # Обновление прогресса
                new_percent = int((current / total_counter) * 100)
                if new_percent > progress_percent:
                    progress_percent = new_percent
                    progress_recorder.set_progress(progress_percent, 100, description=progress_description)
    finally:
        cleanup_driver_pool()  # Закрываем все драйверы

    return "Обработка завершена"


@shared_task(bind=True)
def test_single_url(self,
                    url: str,
                    parser: TestParser,
                    author_id: int,
                    webdriver_options: list[str] = None) -> str:
    """
    Celery-задача для тестирования одного URL с использованием Selenium и парсера.

    :param self: Объект задачи Celery.
    :param url: URL для обработки.
    :param author_id: ID автора (User).
    :param parser: объект класса TestParser
    :param webdriver_options: Опции для Selenium-драйвера.
    :return: Словарь с результатами обработки.
    """
    # Инициализация прогресса
    progress_recorder = ProgressRecorder(self)
    progress_recorder.set_progress(0, 100, description="Начало обработки URL")

    parser_cls_name = parser.class_name
    parser_config = parser.config
    test_parse_report = parser.testparsereport

    parser_dispatcher = WebParserDispatcher()
    parser_cls = parser_dispatcher.get_by_class_name(parser_cls_name)
    parser = parser_cls(config=parser_config if parser_config else {})

    selenium = SeleniumDriver(options=webdriver_options if webdriver_options else None)
    driver = selenium.get_driver()

    result = {
        "url": url,
        "status": None,
        "html": None,
        "parsed_data": None,
        "error": None
    }

    try:
        progress_recorder.set_progress(50, 100, description="Страница загружается")
        fetch_result = fetch_page_with_selenium(url, driver)

        result["status"] = fetch_result["status"]
        result["html"] = fetch_result["html"]

        if fetch_result["html"]:
            progress_recorder.set_progress(75, 100, description="Парсинг данных")
            try:
                parser_result = parser.parse_html(url=url, html=fetch_result["html"])
                result["parsed_data"] = parser_result
            except Exception as e:
                result["error"] = f"Parsing failed: {str(e)}"
                logger.error(f"Parsing failed for {url}: {e}")
        else:
            result["error"] = "No HTML content retrieved"

        progress_recorder.set_progress(100, 100, description="Обработка завершена")
        # Сохранение результата в базе данных
        test_parse_report.status = result["status"]
        test_parse_report.html = result["html"]
        test_parse_report.parsed_data = result["parsed_data"]
        test_parse_report.error = result["error"]
        test_parse_report.save()

    except Exception as e:
        result["error"] = f"Failed to fetch page: {str(e)}"
        logger.error(f"Failed to process {url}: {e}")
        progress_recorder.set_progress(100, 100, description="Обработка завершена с ошибкой")
        test_parse_report.error = result["error"]
        test_parse_report.save()

    finally:
        driver.quit()  # Закрытие Selenium-драйвера

    return "Обработка завершена"
