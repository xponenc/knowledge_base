from typing import Type, Dict
from .base import ContentRecognizer

recognizer_registry: Dict[str, Type[ContentRecognizer]] = {}

def register_recognizer(recognizer_class: Type[ContentRecognizer]):
    for ext in recognizer_class.supported_extensions:
        recognizer_registry.setdefault(ext.lower(), []).append(recognizer_class)

def get_recognizer_for_extension(ext: str, file_path: str) -> ContentRecognizer:
    ext = ext.lower()
    if ext not in recognizer_registry:
        raise ValueError(f"Нет зарегистрированного обработчика для расширения '{ext}'")

    # Пока просто возвращаем первый зарегистрированный
    recognizer_class = recognizer_registry[ext][0]
    return recognizer_class(file_path)
