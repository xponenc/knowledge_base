import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from selenium.webdriver.common.by import By

from app_parsers.services.parsers.base import BaseWebParser, get_parser_class_by_name
from app_parsers.services.parsers.core_parcer_engine import SeleniumDriver

logger = logging.getLogger(__name__)

CONCURRENCY_LIMIT = 5
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)


@shared_task(bind=True)
def parse_urls_task(self,
                    urls: list,
                    parser_cls_name: str,
                    webdriver_options: list[str] = None):

    selenium = SeleniumDriver(options=webdriver_options)

    # Инициализируем парсер по имени
    parser_cls = get_parser_class_by_name(parser_cls_name)
    parser = parser_cls(config={})

    total = len(urls)
    total_counter = len(urls)
    if total_counter == 0:
        return "Обработка завершена: документы не найдены"

    # Инициализация прогресса
    progress_recorder = ProgressRecorder(self)
    progress_description = f'Обрабатывается {total_counter} объектов'
    progress_percent = 0

    current = 0
    results = []

    def fetch_page_with_selenium(url: str) -> Dict[str, Any]:
        """
        Загружает страницу с использованием Selenium и извлекает html.

        :param url: URL страницы для загрузки.
        :return: Словарь с результатами загрузки или ошибкой.
        """
        logger.info(f"page loading started {url}")
        driver = selenium.get_driver()

        try:
            driver.get(url)  # Загрузка страницы
            driver.execute_script("return document.readyState")  # Ожидание полной загрузки страницы
            html = driver.page_source  # Получение HTML-контента страницы

            return {
                "url": url,
                "status": 200,
                "html": html,
            }
        except Exception as e:
            # Обработка ошибок при загрузке страницы
            logger.error(f"page loading failed  {url}: {e}")
            return {
                "url": url,
                "status": None,
                "html": None,
            }
        finally:
            driver.quit()  # Закрытие Selenium-драйвера

    with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
        futures = {executor.submit(fetch_page_with_selenium, url): url for url in urls}
        for i, future in enumerate(as_completed(futures)):
            current += 1
            result = future.result()

            # отправляем результат в парсер

            # Обновление прогресса по процентам
            new_percent = int((current / total_counter) * 100)
            if new_percent > progress_percent:
                progress_percent = new_percent
                progress_recorder.set_progress(progress_percent, 100, description=progress_description)

    return "Обработка завершена"
