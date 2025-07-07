import tiktoken
from typing import Dict, List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app_chunks.splitters.base import BaseSplitter


class SimpleRecursiveSplitter(BaseSplitter):
    """
    Простой рекурсивный сплиттер, разбивает текст на чанки
    по количеству токенов с заданным перекрытием.
    """
    config_schema = {
        "encoding_name": {
            "type": str,
            "label": "Модель разбиения на токены",
            "help_text": '''"cl100k_base" - gpt-4, gpt-4-turbo, gpt-3.5-turbo, text-embedding-ada-002
                            "p50k_base" - code-davinci-002, text-davinci-002, text-davinci-003
                            "p50k_edit" - text-davinci-edit-001, code-davinci-edit-001
                            "r50k_base" - davinci, curie, babbage, ada
                            ''',
        },
        "chunk_size": {
            "type": int,
            "label": "Максимальная длина чанка (в токенах)",
            "help_text": "Максимальная длина чанка (в токенах)",
        },
        "chunk_overlap": {
            "type": int,
            "label": "Размер перекрытия",
            "help_text": "Размер перекрытия чанков (в токенах)",
        },
    }

    name = "Простой рекурсивный"
    help_text = "Простой токен-базированный рекурсивный сплиттер с перекрытием"

    def __init__(self, config: Dict):
        super().__init__(config)

    def split(self, metadata: dict, text_to_split: str) -> List[Document]:
        encoding_name = self.config.get("encoding_name")
        chunk_size = self.config.get("chunk_size")
        chunk_overlap = self.config.get("chunk_overlap")

        if not encoding_name:
            raise ValueError("Не указано имя кодировки (encoding_name)")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: self._num_tokens_from_string(x, encoding_name)
        )

        return [
            Document(page_content=chunk, metadata=metadata.copy())
            for chunk in splitter.split_text(text_to_split)
        ]

    @staticmethod
    def _num_tokens_from_string(string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))
