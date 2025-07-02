"""
Скрипт для загрузки и локального кеширования необходимых моделей трансформеров.

Скрипт проверяет наличие модели в локальном кеше перед загрузкой,
чтобы избежать повторных скачиваний. Это удобно для оффлайн и продакшн окружений.

Модели сохраняются в локальную папку для повторного использования.
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import logging

# Конфигурация логгера
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Папка для локального хранения моделей
CUSTOM_CACHE_DIR = Path(__file__).resolve().parent.parent / "local_models"
CUSTOM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Список моделей для загрузки
MODELS = [
    "ai-forever/FRIDA",
    # "ai-forever/sbert_large_nlu_ru",
    # "intfloat/multilingual-e5-base",
    "DiTy/cross-encoder-russian-msmarco"
]

def is_model_cached(model_id: str, cache_dir: Path) -> bool:
    """
    Проверяет, сохранена ли модель локально.

    Args:
        model_id (str): Идентификатор модели на Hugging Face.
        cache_dir (Path): Папка с кешем моделей.

    Returns:
        bool: True если модель найдена локально, иначе False.
    """
    model_cache_path = cache_dir / f"models--{model_id.replace('/', '--')}"
    return model_cache_path.exists() and any(model_cache_path.glob("**/*"))

def download_model(model_id: str, cache_dir: Path):
    """
    Загружает и сохраняет модель локально.

    Args:
        model_id (str): Идентификатор модели.
        cache_dir (Path): Папка для кеша.

    Raises:
        Exception: В случае ошибки загрузки.
    """
    if "cross-encoder" in model_id:
        CrossEncoder(model_id, cache_folder=str(cache_dir))
    else:
        SentenceTransformer(model_id, cache_folder=str(cache_dir))
        AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
        AutoModel.from_pretrained(model_id, cache_dir=str(cache_dir))

def main():
    """
    Основная функция - проверяет и скачивает модели при необходимости.
    """
    for model_id in MODELS:
        logging.info(f"Checking model: {model_id}")
        if is_model_cached(model_id, CUSTOM_CACHE_DIR):
            logging.info(f"Model already cached: {model_id}")
            continue

        try:
            logging.info(f"Downloading model: {model_id}")
            download_model(model_id, CUSTOM_CACHE_DIR)
            logging.info(f"Successfully downloaded: {model_id}")
        except Exception as e:
            logging.error(f"Failed to download {model_id}: {e}")

    logging.info("Model download check completed.")

if __name__ == "__main__":
    main()
