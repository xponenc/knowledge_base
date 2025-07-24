import platform

import pdfplumber
from pdf2image import convert_from_path
import pytesseract

from recognizers.base import ContentRecognizer
from utils.quality_control.process_quality import evaluate_text_quality
from utils.setup_logger import setup_logger

logger = setup_logger(__name__, log_dir="logs/documents_recognize", log_file="recognizers.log")

# Настройки OCR
"""TESSERACT_CMD Путь к исполняемому файлу Tesseract OCR."""
if platform.system() == "Windows":
    TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # Для Unix предполагаем, что tesseract доступен в PATH
    TESSERACT_CMD = "tesseract"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

OCR_LANGUAGES = 'rus+eng'
"""Языки для распознавания текста в Tesseract OCR (русский и английский)."""

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

"""POPPLER_PATH Путь к библиотеке Poppler для pdf2image."""
if platform.system() == "Windows":
    POPPLER_PATH = r'c:\poppler-24.08.0\Library\bin'
else:
    POPPLER_PATH = "/usr/bin"     # или другой путь, где установлен poppler на Unix


class OCRPDFRecognizer(ContentRecognizer):
    supported_extensions = [".pdf"]

    def recognize(self):
        try:
            images = convert_from_path(self.file_path, dpi=300, poppler_path=POPPLER_PATH)
            ocr_text = "\n".join(pytesseract.image_to_string(img, lang=OCR_LANGUAGES) for img in images)
            logger.info(f"Текст извлечён OCR: {self.file_path}")
            return {
                "text": ocr_text.strip(),
                "method": f"{self.__class__.__name__}(pytesseract):success",
                "quality_report": evaluate_text_quality(text=ocr_text)
            }
        except Exception as e:
            logger.error(f"OCR не справился: {e}")
            return {
                "text": "",
                "method": f"{self.__class__.__name__}(pytesseract):failed",
                "quality_report": {}
            }
