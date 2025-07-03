import pypdf

from recognizers.base import ContentRecognizer
from utils.quality_control.process_quality import evaluate_text_quality
from utils.setup_logger import setup_logger

logger = setup_logger(__name__, log_dir="logs/documents_recognize", log_file="recognizers.log")


class PyPDFRecognizer(ContentRecognizer):
    """
    Класс для распознавания текста из PDF-файлов.

    Наследуется от базового класса ContentRecognizer и использует библиотеку pypdf
    для извлечения текста из файлов с расширениями .pdf.

    Attributes:
        supported_extensions (list): Список поддерживаемых расширений файлов ([".pdf"]).
    """

    supported_extensions = [".pdf"]

    def recognize(self):
        """
        Извлекает текст из PDF-файла и оценивает его качество.

        Returns:
            dict: Словарь с результатами обработки, содержащий:
                - text (str): Извлечённый текст из PDF.
                - method (str): Информация об успехе или неудаче обработки (например, "PDFRecognizer:success").
                - quality_report (dict): Результат оценки качества текста, полученный из evaluate_text_quality.

        Raises:
            Exception: Если при обработке PDF-файла возникает ошибка, логируется ошибка,
                       и возвращается словарь с пустым текстом и статусом неудачи.
        """
        try:
            # Открываем PDF-файл
            with open(self.file_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                # Извлекаем текст со всех страниц
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                # Удаляем лишние пробелы и переносы строк
                text = text.strip()
                logger.info(f"Текст извлечён из PDF: {self.file_path}")
                return {
                    "text": text,
                    "method": f"{self.__class__.__name__}:success",
                    "quality_report": evaluate_text_quality(text=text)
                }
        except Exception as e:
            logger.error(f"Ошибка при обработке PDF: {e}")
            return {
                "text": "",
                "method": f"{self.__class__.__name__}:failed",
                "quality_report": {}
            }
