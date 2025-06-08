import abc
from typing import Dict


class BaseWebParser(abc.ABC):
    """
       Базовый класс для всех веб-парсеров.

       Атрибуты:
           config_schema (dict): Схема конфигурации парсера.
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

    def __init__(self, config: Dict):
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'config_schema') or not isinstance(cls.config_schema, dict):
            raise TypeError(
                f"Класс {cls.__name__} должен определить атрибут `config_schema` как словарь"
            )

    @abc.abstractmethod
    def parse_html(self, html) -> Dict:
        """
        Основной метод, который должен возвращать очищенный контент для html:
        {
            "title": str, # заголовок основного контента страницы
            "tags": list[str], # список тэгов (категорий) страницы
            "content": str, # основной контент страницы
            "metadata": dict # словарь с метаданными страницы
        }
        """
        pass
