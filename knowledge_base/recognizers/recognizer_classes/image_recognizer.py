import pytesseract


from recognizers.base import ContentRecognizer
from utils.quality_control.process_quality import evaluate_text_quality
from utils.setup_logger import setup_logger

logger = setup_logger(__name__, log_dir="logs/documents_recognize", log_file="recognizers.log")

# Настройки OCR
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
"""Путь к исполняемому файлу Tesseract OCR."""

OCR_LANGUAGES = 'rus+eng'
"""Языки для распознавания текста в Tesseract OCR (русский и английский)."""


class ImageRecognizer(ContentRecognizer):
    supported_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]

    def recognize(self):
        try:
            text = pytesseract.image_to_string(self.file_path, lang=OCR_LANGUAGES)
            logger.info(f"Текст распознан OCR из изображения: {self.file_path}")
            return {
                "text": text,
                "method": "image_ocr",
                "quality_report": evaluate_text_quality(text=text)
            }
        except Exception as e:
            logger.error(f"Ошибка OCR при обработке изображения: {e}")
            return {
                "text": "",
                "method": "image_ocr_failed",
                "quality_report": {}
            }
