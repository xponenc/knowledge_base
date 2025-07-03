from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table


from recognizers.base import ContentRecognizer
from utils.quality_control.process_quality import evaluate_text_quality
from utils.setup_logger import setup_logger

logger = setup_logger(__name__, log_dir="logs/documents_recognize", log_file="recognizers.log")


class DOCXRecognizer(ContentRecognizer):
    supported_extensions = [".docx", ".doc", ]

    def recognize(self):
        try:
            doc = Document(self.file_path)
            result = ""

            for element in doc.element.body:
                if isinstance(element, CT_P):
                    paragraph = Paragraph(element, doc)
                    text = paragraph.text.strip()
                    if text:
                        result += f"{text}\n"
                elif isinstance(element, CT_Tbl):
                    table = Table(element, doc)
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        result += " | ".join(cells) + "\n"

            logger.info(f"Текст извлечён из DOCX: {self.file_path}")
            return {
                "text": result.strip(),
                "method": f"{self.__class__.__name__}:success",
                "quality_report": evaluate_text_quality(text=result)
            }
        except Exception as e:
            logger.error(f"Ошибка при обработке DOCX: {e}")
            return {
                "text": "",
                "method": f"{self.__class__.__name__}:failed",
                "quality_report": {}
            }
