from typing import Dict, Type

from app_parsers.services.parsers.base import BaseWebParser

WEB_PARSER_REGISTRY: Dict[str, Type[BaseWebParser]] = {}


def register_parser(parser_class: Type[BaseWebParser]):

    name = parser_class.__name__
    module = parser_class.__module__
    full_name = f"{module}.{name}"
    if full_name in WEB_PARSER_REGISTRY:
        raise ValueError(f"Парсер {full_name} уже зарегистрирован")
    WEB_PARSER_REGISTRY[full_name] = parser_class


def get_parser_class_by_name(parser_class_name: str) -> Type[BaseWebParser]:
    if parser_class_name not in WEB_PARSER_REGISTRY:
        raise ValueError(f"Парсер с именем {parser_class_name} не найден в реестре")
    return WEB_PARSER_REGISTRY[parser_class_name]
