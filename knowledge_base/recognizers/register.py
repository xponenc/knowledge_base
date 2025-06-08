from typing import Dict, Type

from recognizers.base import ContentRecognizer

RECOGNIZER_REGISTRY: Dict[str, Type[ContentRecognizer]] = {}


def register_recognizer(recognizer_class: Type[ContentRecognizer]):
    for ext in recognizer_class.supported_extensions:
        RECOGNIZER_REGISTRY.setdefault(ext.lower(), []).append(recognizer_class)


def get_recognizer_for_extension(ext: str, file_path: str) -> ContentRecognizer:
    ext = ext.lower()
    if ext not in RECOGNIZER_REGISTRY:
        raise ValueError(f"Нет зарегистрированного обработчика для расширения '{ext}'")

    # Пока просто возвращаем первый зарегистрированный
    recognizer_class = RECOGNIZER_REGISTRY[ext][0]
    return recognizer_class(file_path)