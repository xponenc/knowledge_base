import pdfplumber
from pdf2image import convert_from_path
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

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

POPPLER_PATH = r'c:\poppler-24.08.0\Library\bin'
"""Путь к библиотеке Poppler для pdf2image."""


class PDFRecognizer(ContentRecognizer):
    supported_extensions = [".pdf"]

    def recognize(self):
        try:
            with pdfplumber.open(self.file_path) as pdf:
                all_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        all_text += text + "\n"
                if all_text.strip():
                    logger.info(f"Текст извлечён pdfplumber: {self.file_path}")
                    return {
                        "text": all_text,
                        "method": f"{self.__class__.__name__}(pdfplumber):failed",
                        "quality_report": evaluate_text_quality(text=all_text)
                    }
        except Exception as e:
            logger.warning(f"pdfplumber не справился: {e}")

        try:
            images = convert_from_path(self.file_path, dpi=300, poppler_path=POPPLER_PATH)
            ocr_text = "\n".join(pytesseract.image_to_string(img, lang=OCR_LANGUAGES) for img in images)
            logger.info(f"Текст извлечён OCR: {self.file_path}")
            return {
                "text": ocr_text,
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
