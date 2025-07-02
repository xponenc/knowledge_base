import logging
from typing import Dict, Type

from app_parsers.services.parsers.base import BaseWebParser

logger = logging.getLogger(__name__)

WEB_PARSER_REGISTRY: Dict[str, Type[BaseWebParser]] = {}


def register_parser(parser_cls: Type[BaseWebParser]) -> None:
    """
    Регистрирует класс парсера в реестре.

    Args:
        parser_cls: Класс парсера, наследник BaseWebParser.
    """
    parser_name = f"{parser_cls.__module__}.{parser_cls.__name__}"
    if parser_name not in WEB_PARSER_REGISTRY:
        WEB_PARSER_REGISTRY[parser_name] = parser_cls
        logger.debug(f"Parser {parser_name} successfully registered")
    else:
        logger.debug(f"Parser {parser_name} is already registered, skipping")


def get_parser_class_by_name(parser_class_name: str) -> Type[BaseWebParser]:
    if parser_class_name not in WEB_PARSER_REGISTRY:
        raise ValueError(f"Парсер с именем {parser_class_name} не найден в реестре")
    return WEB_PARSER_REGISTRY[parser_class_name]
