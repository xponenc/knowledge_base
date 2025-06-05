import pandas as pd

from recognizers.base import ContentRecognizer
from utils.quality_control.process_quality import evaluate_text_quality
from utils.setup_logger import setup_logger

logger = setup_logger(__name__, log_dir="logs/documents_recognize", log_file="recognizers.log")


class ExcelRecognizer(ContentRecognizer):
    supported_extensions = [".xls", ".xlsx"]

    def recognize(self):
        try:
            df = pd.read_excel(self.file_path, engine="openpyxl")
            text = df.to_string(index=False)
            logger.info(f"Текст извлечён из Excel: {self.file_path}")
            return {
                "text": text,
                "method": f"{self.__class__.__name__}:success",
                "quality_report": evaluate_text_quality(text=text)
            }
        except Exception as e:
            logger.error(f"Ошибка при обработке Excel: {e}")
            return {
                "text": "",
                "method": f"{self.__class__.__name__}:failed",
                "quality_report": {}
            }
