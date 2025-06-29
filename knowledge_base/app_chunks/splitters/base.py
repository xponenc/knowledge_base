import abc
from typing import Dict, List

from langchain_core.documents import Document


class BaseSplitter(abc.ABC):
    """
       Базовый класс сплиттер.

       Атрибуты:
           config_schema (dict): Схема конфигурации сплиттера.
               Должна быть определена в каждом наследуемом классе и описывать структуру конфигурации.
               Формат:
               {
                   "ключ_конфига": {
                       "type": list[str],  # или другой ожидаемый тип
                       "label": "Название поля",  # отображаемое имя
                       "help_text": "Подсказка пользователю",  # пояснение
                   },
                   ...
               }

               Пример:
               {
                   "exclude_classes": {
                       "type": list[str],
                       "label": "Исключаемые CSS классы",
                       "help_text": "Вводите названия CSS классов, по одному на строку",
                       "required": True,
                   }
               }
   """

    config_schema: Dict = {}
    name: str = ""
    help_text: str = ""

    def __init__(self, config: Dict):
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'config_schema') or not isinstance(cls.config_schema, dict):
            raise TypeError(
                f"Класс {cls.__name__} должен определить атрибут `config_schema` как словарь"
            )

    @abc.abstractmethod
    def split(self, metadata: dict, text_to_split: str) -> List[Document]:
        """
        metadata: dict - словарь с исходными метаданными, необходимыми для формирования итоговых metadata
        text_to_spit: str -  исходный текст для разбиения
        Основной метод, который должен возвращать список Document
        """
        pass
