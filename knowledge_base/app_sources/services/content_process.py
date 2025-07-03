import logging

from django.core.files.base import ContentFile

from app_sources.content_models import RawContent, CleanedContent
from app_sources.services.summary import summarize_text
from recognizers.dispatcher import ContentRecognizerDispatcher

logger = logging.getLogger(__name__)


def recognize_raw_content(raw_content: RawContent) -> dict:
    """
    Пытается распознать текст из raw_content с помощью всех доступных recognizer'ов.
    Возвращает dict с полями: text, method, quality_report
    """
    dispatcher = ContentRecognizerDispatcher()
    file_extension = raw_content.file_extension()
    recognizers = dispatcher.get_recognizers_for_extension(file_extension)

    if not recognizers:
        raise ValueError(f"Нет распознавателей для расширения {file_extension}")

    for recognizer_class in recognizers:
        recognizer = recognizer_class(raw_content.file.path)
        report = recognizer.recognize()

        text = report.get("text", "")
        if text and text.strip():
            return {
                "text": text,
                "method": report.get("method", recognizer_class.__name__),
                "quality_report": report.get("quality_report", {})
            }

    raise ValueError("Не удалось распознать текст ни одним из распознавателей.")


def summarize_and_save_to_document(text: str, document, mode: str = "summary big"):
    """Самаризация текста"""
    summary = summarize_text(text, mode=mode)
    if summary and summary.strip():
        document.description = summary
        document.save(update_fields=["description"])
    else:
        logger.warning(f"Пустой или некорректный результат саммаризации для документа id {document.id}")


def recognize_and_summarize_content(raw_content: RawContent, user_id=None, do_summarization: bool = False):
    """
    Распознаёт текст из RawContent, создаёт CleanedContent и при необходимости выполняет саммаризацию.

    Args:
        raw_content (RawContent): объект RawContent
        user_id (int | None): ID пользователя (автор CleanedContent)
        do_summarization (bool): выполнять ли саммаризацию

    Raises:
        Exception: если нет распознавателей
        ValueError: если текст не распознан
    """
    doc_id = raw_content.network_document.id
    raw_id = raw_content.id
    try:
        logger.info(f"[RawContent {raw_id} / Document {doc_id}] Запуск распознавания")

        result = recognize_raw_content(raw_content)
        text = result["text"]
        method = result["method"]
        quality = result["quality_report"]

        logger.info(f"[RawContent {raw_id} / Document {doc_id}] Текст успешно распознан методом {method}")

        # Удаляем старое CleanedContent
        CleanedContent.objects.filter(raw_content=raw_content).delete()

        # Сохраняем CleanedContent
        cleaned = CleanedContent.objects.create(
            network_document=raw_content.network_document,
            raw_content=raw_content,
            recognition_method=method,
            recognition_quality=quality,
            preview=text[:200],
            author_id=user_id,
        )
        cleaned.file.save("cleaned.txt", ContentFile(text.encode("utf-8")))
        cleaned.save()

        logger.info(f"[RawContent {raw_id} / Document {doc_id}] CleanedContent создан (ID {cleaned.id})")

        # Саммаризация, если нужно
        if do_summarization:
            summarize_and_save_to_document(text, raw_content.network_document)
            logger.info(f"[RawContent {raw_id} / Document {doc_id}] Саммаризация выполнена")

    except Exception as e:
        logger.error(f"[RawContent {raw_id} / Document {doc_id}] Ошибка при обработке: {e}")
        raise

