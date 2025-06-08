import os
import logging
from typing import List, Type, Dict, Any

from .base import ContentRecognizer
from utils.setup_logger import setup_logger
from .register import RECOGNIZER_REGISTRY, get_recognizer_for_extension

logger = setup_logger(__name__, log_dir="logs/documents_recognize", log_file="recognizers.log")


class ContentRecognizerDispatcher:
    def get_by_name(self, name: str) -> Type[ContentRecognizer]:
        """
        Возвращает класс-распознаватель по его имени.

        Args:
            name (str): Имя класса-распознавателя.

        Returns:
            Type[ContentRecognizer]: Найденный класс.

        Raises:
            ValueError: Если не найден распознаватель с указанным именем.
        """
        for recognizer_cls in self._discover_recognizers():
            if recognizer_cls.__name__ == name:
                return recognizer_cls
        raise ValueError(f"Распознаватель с именем '{name}' не найден.")


    def recognize_file(self, file_path: str) -> Dict[str, Any]:
        """
        Распознаёт файл, используя первый подходящий распознаватель.

        Args:
            file_path (str): Путь к файлу.

        Returns:
            dict: Результат распознавания.

        Raises:
            ValueError: Если не найден подходящий распознаватель.
        """
        ext = os.path.splitext(file_path)[1].lower()
        try:
            recognizer = get_recognizer_for_extension(ext, file_path)
            return recognizer.recognize(file_path)
        except ValueError as e:
            logger.error(f"Не найден распознаватель для расширения '{ext}': {e}")
            return {
                "text": "",
                "method": "unsupported",
                "quality_report": {},
                "error": str(e),
            }

    def get_recognizers_for_extension(self, extension: str) -> List[Type[ContentRecognizer]]:
        """
        Возвращает список всех распознавателей, поддерживающих указанное расширение.

        Args:
            extension (str): Расширение файла (например, '.pdf').

        Returns:
            list: Классы распознавателей.
        """
        matching = []
        for recognizer_cls in self._discover_recognizers():
            if extension.lower() in recognizer_cls.supported_extensions:
                matching.append(recognizer_cls)
        return matching

    def recognize_with(self, recognizer_cls: Type[ContentRecognizer], file_path: str) -> Dict[str, Any]:
        """
        Применяет конкретный класс-распознаватель для файла.

        Args:
            recognizer_cls: Класс-распознаватель.
            file_path: Путь к файлу.

        Returns:
            dict: Результат распознавания.

        Raises:
            TypeError: Если передан не тот класс.
            ValueError: Если расширение файла не поддерживается этим классом.
        """
        if not issubclass(recognizer_cls, ContentRecognizer):
            raise TypeError("Класс должен наследовать ContentRecognizer")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in recognizer_cls.supported_extensions:
            raise ValueError(f"{recognizer_cls.__name__} не поддерживает расширение {ext}")

        return recognizer_cls().recognize(file_path)

    def _discover_recognizers(self) -> List[Type[ContentRecognizer]]:
        """
        Возвращает все уникальные классы-распознаватели из регистра.
        """
        all_classes = set()
        for classes in RECOGNIZER_REGISTRY.values():
            all_classes.update(classes)
        print(all_classes)
        return list(all_classes)
