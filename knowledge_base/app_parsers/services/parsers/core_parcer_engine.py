import asyncio
import os
import platform
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Dict, Any

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


###
#  Selenium WebDriver на Windows
###
# https://googlechromelabs.github.io/chrome-for-testing/


def get_chromedriver_path():
    # Попытка найти chromedriver в PATH
    path = shutil.which("chromedriver")
    if path:
        return path

    # Если не найден, используем хардкод в зависимости от ОС
    system = platform.system()
    if system == "Windows":
        return "C:/WebDriver/chromedriver.exe"
    elif system == "Linux":
        return "/usr/local/bin/chromedriver"
    elif system == "Darwin":
        return "/usr/local/bin/chromedriver"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")


class SeleniumDriver:
    def __init__(self, options: list[str] = None):
        if options is None:
            options = ["--headless",
                       "--no-sandbox",
                       "--disable-dev-shm-usage",
                       "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
                       "--start-maximized"]
        self.options = options

    def get_driver(self) -> webdriver.Chrome:
        """
        Создаёт и настраивает экземпляр Chrome WebDriver.

        :return: Объект WebDriver
        """
        options = Options()
        for option in self.options:
            options.add_argument(option)

        # chromedriver_path = get_chromedriver_path()
        # service = Service(executable_path=chromedriver_path)
        # ChromeDriverManager автоматически скачает и настроит chromedriver
        # path = ChromeDriverManager().install()
        service = Service(ChromeDriverManager().install())

        return webdriver.Chrome(service=service, options=options)

    @contextmanager
    def driver(self):
        driver = self.get_driver()
        try:
            yield driver
        finally:
            try:
                driver.quit()
            except Exception:
                pass