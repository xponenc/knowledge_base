from typing import Dict, Type

from app_chunks.splitters.base import BaseSplitter

CHUNK_SPLITTER_REGISTRY: Dict[str, Type[BaseSplitter]] = {}


def register_splitter(splitter_class: Type[BaseSplitter]):
    """
        Регистрирует класс сплиттера в глобальном реестре.
        Использует полное имя класса (модуль + имя) как ключ для уникальности.

        Args:
            splitter_class (Type[BaseSplitter]): Класс сплиттера для регистрации.

        Raises:
            ValueError: Если сплиттер с таким именем уже зарегистрирован.
        """

    name = splitter_class.__name__
    module = splitter_class.__module__
    full_name = f"{module}.{name}"
    if full_name in CHUNK_SPLITTER_REGISTRY:
        raise ValueError(f"Сплиттер {full_name} уже зарегистрирован")
    CHUNK_SPLITTER_REGISTRY[full_name] = splitter_class

#
# def get_parser_class_by_name(parser_class_name: str) -> Type[BaseWebParser]:
#     if parser_class_name not in WEB_PARSER_REGISTRY:
#         raise ValueError(f"Парсер с именем {parser_class_name} не найден в реестре")
#     return WEB_PARSER_REGISTRY[parser_class_name]