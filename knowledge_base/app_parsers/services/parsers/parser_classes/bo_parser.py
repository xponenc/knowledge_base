from typing import Dict

from app_parsers.services.parsers.base import BaseWebParser


class BOWebParser(BaseWebParser):
    config_schema = {
        "exclude_classes": {
            "type": list[str],
            "label": "Исключаемые CSS классы",
            "help_text": "Вводите названия CSS классов, по одному на строку"
        },
        "exclude_tags": {
            "type": list[str],
            "label": "Исключаемые HTML теги",
            "help_text": "Вводите названия тегов, по одному на строку (например: script, style)"
        },
    }

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

    def parse_html(self, html: str) -> Dict:
        """
        Основной метод, который должен возвращать:
        {
            "title": str, # заголовок основного контента страницы
            "tags": list[str], # список тэгов (категорий) страницы
            "content": str, # основной контент страницы
            "metadata": dict # словарь с метаданными страницы
        }
        """
        pass
