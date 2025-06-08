import abc
from typing import List, Type, Dict


class ContentRecognizer(abc.ABC):
    supported_extensions: List[str] = []

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abc.abstractmethod
    def recognize(self) -> Dict:
        """
        Основной метод, который должен возвращать:
        {
            "text": str,
            "method": str,
            "quality_report": dict
        }
        """
        pass
