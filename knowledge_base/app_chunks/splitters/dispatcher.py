from typing import Type, List

from app_chunks.splitters.base import BaseSplitter
from app_chunks.splitters.registry import CHUNK_SPLITTER_REGISTRY
from utils.setup_logger import setup_logger

logger = setup_logger(__name__, log_dir="logs/chunking", log_file="chunking.log")


class SplitterDispatcher:
    @classmethod
    def get_by_name(cls, name: str) -> Type[BaseSplitter]:
        """
        Возвращает класс-сплиттер по его имени.

        Args:
            name (str): Имя класса-сплиттера.

        Returns:
            Type[BaseSplitter]: Найденный класс.

        Raises:
            ValueError: Если не найден распознаватель с указанным именем.
        """

        splitter_cls = CHUNK_SPLITTER_REGISTRY.get(name)

        if splitter_cls:
            return splitter_cls
        logger.error(f"Сплиттер с именем '{name}' не найден.")
        raise ValueError(f"Сплиттер с именем '{name}' не найден.")

    @classmethod
    def list_all(cls) -> list[Type[BaseSplitter]]:
        """
        Возвращает список всех зарегистрированных имён сплиттеров.

        Returns:
            List[str]: Список имён сплиттеров.
        """
        return list(CHUNK_SPLITTER_REGISTRY.values())

    @classmethod
    def create_instance(cls, name: str, config: dict) -> BaseSplitter:
        """
        Создаёт экземпляр сплиттера по имени с заданной конфигурацией.

        Args:
            name (str): Имя класса сплиттера.
            config (dict): Конфигурационный словарь для инициализации сплиттера.

        Returns:
            BaseSplitter: Экземпляр сплиттера.

        Raises:
            ValueError: Если сплиттер с указанным именем не найден.
        """
        splitter_cls = cls.get_by_name(name)
        return splitter_cls(config)
