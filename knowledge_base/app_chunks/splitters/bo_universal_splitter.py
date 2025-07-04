import re
from typing import Dict, List

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app_chunks.splitters.base import BaseSplitter


class BoHybridMarkdownSplitter(BaseSplitter):
    config_schema = {
        "encoding_name": {
            "type": str,
            "label": "Модель разбиения на токены",  # отображаемое имя
            "help_text": '''"cl100k_base" -     gpt-4, gpt-4-turbo, gpt-3.5-turbo, text-embedding-ada-002\n
            "p50k_base" -   code-davinci-002, text-davinci-002, text-davinci-003\n
            "p50k_edit" -	text-davinci-edit-001, code-davinci-edit-001\n
            "r50k_base" -	davinci, curie, babbage, ada"''',  # пояснение
        },

        "chunk_size": {
            "type": int,  # или другой ожидаемый тип
            "label": "Максимальная длина чанка (в токенах)",  # отображаемое имя
            "help_text": "Максимальная длина чанка (в токенах), к чанкам с длиной более заданной "
                         "будет применено дополнительно рекурсивное разбиение до заданного размера",  # пояснение
        },
        "chunk_overlap": {
            "type": int,  # или другой ожидаемый тип
            "label": "Размер перекрытия",  # отображаемое имя
            "help_text": "Размер перекрытия чанков при рекурсивном разбиении (в токенах)",  # пояснение
        },
        "header_levels": {
            "type": list[int],
            "label": "Уровни заголовков Markdown",
            "help_text": "Уровни Markdown-заголовков, по которым нужно разбивать (например: 1, 2, 3)",
        },
        "min_tail": {
            "type": int,
            "label": "Минимальный размер последнего чанка (в процентах)",
             "help_text": (
                "Если размер последнего чанка составляет менее указанного процента от 'Максимальной длины чанка', "
                "то он будет присоединён к предыдущему чанку.\n"
                "Например, при значении 15 и размере чанка 1000 токенов — хвост короче 150 токенов объединяется с предыдущим."
            ),
        },
    }
    name = "Маркдаун + рекурсив"
    help_text = ("Разделяет текст по маркдаун разметке, после чего анализируется размер чанка и если он превышает "
                 "заданный, то выполняется дополнительная разбивка чанка рекурсивным сплиттером")

    def __init__(self, config: Dict):
        super().__init__(config)

    def split(self, metadata: dict, text_to_split: str) -> List[Document]:
        print(f"{self.config=}")
        total_header = self._generate_total_header_from_metadata(metadata=metadata)

        source_chunks = self._split_text(text_to_split,
                                                    chunk_size=self.config.get("chunk_size"),
                                                    chunk_overlap=self.config.get("chunk_overlap")
                                                    )
        for chunk in source_chunks:
            chunk.metadata.update(metadata)

        source_chunks = [self._process_document(chunk) for chunk in source_chunks]

        for chunk in source_chunks:
            chunk.page_content = total_header + chunk.page_content
            chunk.metadata = self._flatten_metadata(chunk.metadata)

        return source_chunks

    @classmethod
    def _generate_total_header_from_metadata(cls, metadata: dict) -> str:
        """
        Генерирует строку `total_header` на основе метаданных документа.

        - Если есть `title`, добавляет строку "Заголовок: {title}".
        - Если `tags` содержит больше одного элемента:
            - Удаляет первый элемент.
            - Исключает все теги, равные `title`.
            - Если после фильтрации остались теги — добавляет строку "Категории: {tag1, tag2, ...}".

        Args:
            metadata (dict): Словарь метаданных, содержащий ключи "title" и "tags".

        Returns:
            str: Сформированная строка заголовка.
        """
        total_header = ""

        title = metadata.get("title")

        tags = metadata.get("tags") or []
        if isinstance(tags, list) and len(tags) > 1:
            filtered_tags = tags[1:]
            filtered_tags = [tag for tag in filtered_tags if tag != title]
            if filtered_tags:
                total_header += "Категории: " + ", ".join(filtered_tags) + ". "
        if title:
            total_header += f"Заголовок: {title}. "

        return total_header

    @staticmethod
    def _num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Возвращает количество токенов в строке"""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def _split_text(self, text, chunk_size: int, chunk_overlap: int):
        levels = self.config.get("header_levels", [1])  # по умолчанию только #
        headers_to_split_on = [(f"{'#' * level}", f"Header {level}") for level in levels]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        fragments = markdown_splitter.split_text(text)

        encoding_name = self.config.get("encoding_name")
        if not encoding_name:
            raise ValueError("Не указано имя кодировки (`encoding_name`) в конфигурации сплиттера.")

        try:
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=lambda x: self._num_tokens_from_string(x, encoding_name)
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации RecursiveCharacterTextSplitter: {e}") from e

        source_chunks = []

        for fragment in fragments:
            fragment_len = self._num_tokens_from_string(fragment.page_content, encoding_name)
            if fragment_len > chunk_size:
                chunks = recursive_splitter.split_text(fragment.page_content)
                for chunk in chunks:
                    new_document = Document(page_content=chunk, metadata=fragment.metadata)
                    new_document.metadata["size_in_tokens"] = self._num_tokens_from_string(new_document.page_content,
                                                                                           encoding_name)
                    source_chunks.append(new_document)
            else:
                fragment.metadata["size_in_tokens"] = fragment_len
                source_chunks.append(fragment)

        # --- Вспомогательная функция для поиска максимального перекрытия ---
        def find_overlap(a: str, b: str, min_overlap: int = 10) -> int:
            max_len = min(len(a), len(b))
            for length in range(max_len, min_overlap - 1, -1):
                if a[-length:] == b[:length]:
                    return length
            return 0

        # --- Обработка min_tail (слияние последнего короткого чанка с предыдущим с учётом перекрытия) ---
        min_tail_percent = self.config.get("min_tail", 0)
        if min_tail_percent and len(source_chunks) >= 2:
            last_chunk = source_chunks[-1]
            penultimate_chunk = source_chunks[-2]

            threshold_tokens = int(chunk_size * min_tail_percent / 100)
            last_chunk_tokens = last_chunk.metadata.get("size_in_tokens", 0)

            if last_chunk_tokens < threshold_tokens:
                penultimate_text = penultimate_chunk.page_content
                last_text = last_chunk.page_content

                # Ищем длину перекрытия между концом предпоследнего и началом последнего чанков
                overlap_len = find_overlap(penultimate_text, last_text, min_overlap=chunk_overlap)

                # Склеиваем с учётом перекрытия (без дублирования)
                combined_content = penultimate_text + last_text[overlap_len:]

                combined_metadata = penultimate_chunk.metadata.copy()
                combined_metadata["size_in_tokens"] = self._num_tokens_from_string(combined_content, encoding_name)

                combined_document = Document(page_content=combined_content, metadata=combined_metadata)

                source_chunks = source_chunks[:-2] + [combined_document]

        return source_chunks

    @classmethod
    def _clean_metadata(cls, obj):
        """
        Рекурсивно очищает вложенные структуры dict и list от пустых значений:
        - None
        - пустые строки ""
        - пустые списки []
        - пустые словари {}

        :param obj: входной объект (dict, list, или любой другой)
        :return: очищенный объект или None, если он "пустой"
        """
        if isinstance(obj, dict):
            cleaned = {
                k: cls._clean_metadata(v)
                for k, v in obj.items()
                if (v_cleaned := cls._clean_metadata(v)) is not None
            }
            return cleaned or None

        elif isinstance(obj, list):
            cleaned = [cls._clean_metadata(v) for v in obj if (cls._clean_metadata(v) is not None)]
            return cleaned or None

        return obj if obj not in (None, "", [], {}) else None

    @classmethod
    def _process_links(cls, metadata, prefix='internal_links'):
        internal_links = metadata.get(prefix, [])
        metadata.pop(prefix)
        if isinstance(internal_links, list):
            for i, item in enumerate(internal_links):
                if isinstance(item, list) and len(item) == 2:
                    name, url = item
                    metadata[f"{prefix}__{i}"] = f"{name} {url}"
        return metadata

    @classmethod
    def _process_document(cls, doc: Document) -> Document:
        # Извлекаем page_content и metadata из документа
        page_content = doc.page_content
        metadata = doc.metadata.copy()  # Создаём копию metadata
        files = metadata.get('files', {})


        # Рекурсивная функция для обработки вложенных словарей и списков в files
        def process_files(data, page_content, prefix='files', parent_name=None):
            if isinstance(data, dict):
                for key, value in data.items():
                    page_content = process_files(value, page_content, f"{prefix}__{key}", parent_name)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    # Если элемент списка — список, передаём первый элемент как имя
                    if isinstance(item, list):
                        name = item[0] if len(item) > 0 else None
                        for j, sub_item in enumerate(item):
                            page_content = process_files(sub_item, page_content, f"{prefix}__{i}__{j}", name)
                    else:
                        page_content = process_files(item, page_content, f"{prefix}__{i}", parent_name)
            elif isinstance(data, str) and 'http' in data:
                # Проверяем, является ли строка ссылкой
                # Ищем ссылку в page_content в формате Markdown или как чистую строку
                markdown_pattern = rf'!\[(.*?)\]\({re.escape(data)}\)'
                match = re.search(markdown_pattern, page_content)
                if match or data in page_content:
                    # Используем parent_name как название, если оно есть, иначе название из Markdown или ссылку
                    name = parent_name if parent_name else (match.group(1) if match else data)
                    # Добавляем в metadata на корневой уровень
                    metadata[prefix] = f"{name} {data}"
                    # Заменяем Markdown-ссылку на название или чистую ссылку на имя
                    if match:
                        page_content = re.sub(markdown_pattern, match.group(1), page_content)
                    else:
                        page_content = page_content.replace(data, name)
            return page_content


        # Обрабатываем все элементы в files, обновляя page_content
        page_content = process_files(files, page_content)
        # Обработка metadata.internal_links
        metadata = cls._process_links(metadata, prefix='internal_links')
        # Обработка metadata.external_links
        metadata = cls._process_links(metadata, prefix="external_links")

        # Удаляет Markdown-заголовки (###, ##, #) в любом месте строки
        page_content = re.sub(r'\s*#{1,6}\s*', ' ', page_content)
        # Удаляем лишние пробелы, оставшиеся после замены Markdown-ссылок
        page_content = re.sub(r'\s+', ' ', page_content).strip()

        # Устанавливаем files в пустой список
        metadata['files'] = []
        metadata = cls._clean_metadata(metadata)

        # Создаём новый объект Document с обновлёнными данными
        return Document(page_content=page_content, metadata=metadata)

    @classmethod
    def _flatten_metadata(cls, d, parent_key='', sep='_'):
        """
        Рекурсивно разворачивает вложенный словарь/список в плоский словарь.
        Ключи соединяются через sep, списки индексируются.
        """
        items = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(cls._flatten_metadata(v, new_key, sep=sep).items())
        elif isinstance(d, list):
            for i, v in enumerate(d):
                new_key = f"{parent_key}{sep}{i}"
                items.extend(cls._flatten_metadata(v, new_key, sep=sep).items())
        else:
            items.append((parent_key, d))
        return dict(items)

