from typing import List, Type

from .base import BaseWebParser
from utils.setup_logger import setup_logger
from .registry import WEB_PARSER_REGISTRY

logger = setup_logger(__name__, log_dir="logs/documents_parser", log_file="parser.log")


class WebParserDispatcher:
    def get_by_class_name(self, name: str) -> Type[BaseWebParser]:
        """
        Возвращает класс-парсер по его имени.

        Args:
            name (str): Имя класса-парсер.

        Returns:
            Type[BaseWebParser]: Найденный класс.

        Raises:
            ValueError: Если не найден парсер с указанным именем.
        """
        parser_cls = WEB_PARSER_REGISTRY.get(name)
        if parser_cls:
            return parser_cls
        raise ValueError(f"Парсер с именем '{name}' не найден.")

    @staticmethod
    def discover_parsers() -> List[Type[BaseWebParser]]:
        """
        Возвращает все классы-парсеры из регистра.
        """
        all_classes = set()
        for parser_cls in WEB_PARSER_REGISTRY.values():
            all_classes.add(parser_cls)
        return list(all_classes)
