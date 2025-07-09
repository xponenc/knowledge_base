import re
from typing import Dict, List
import tiktoken
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from app_chunks.splitters.base import BaseSplitter


class LegalLangChainSplitter(BaseSplitter):
    config_schema = {
        "encoding_name": {
            "type": str,
            "label": "Модель токенизации",
            "help_text": "cl100k_base для GPT-4/3.5, p50k_base для старых моделей"
        },
        "min_chunk_size": {
            "type": int,
            "label": "Минимальный размер чанка (токены)",
            "help_text": "Базовый минимум, адаптируется к типу документа"
        },
        "max_chunk_size": {
            "type": int,
            "label": "Максимальный размер чанка (токены)",
            "help_text": "Базовый максимум, адаптируется к типу документа"
        },
        "chunk_overlap": {
            "type": int,
            "label": "Перекрытие чанков (токены)",
            "help_text": "Размер перекрытия для связности контекста"
        }
    }
    
    name = "LangChain Legal"
    help_text = "Адаптивный LangChain чанкер для правовых документов (с поддержкой токенов)"
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_tokens = config.get("min_chunk_size", 500)
        self.max_tokens = config.get("max_chunk_size", 1200)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        
        encoding_name = config.get("encoding_name", "cl100k_base")
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        self.recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=min(self.max_tokens, 800),  # Ограничиваем для лучшего распределения
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
            add_start_index=False,
        )
    
    def _count_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def _merge_small_chunks(self, documents: List[Document]) -> List[Document]:
        """Объединяет мелкие чанки для достижения минимального размера."""
        if len(documents) <= 1:
            return documents
        
        optimized_docs = []
        current_doc = None
        
        for doc in documents:
            token_count = doc.metadata['token_count']
            
            # Если чанк меньше минимального размера
            if token_count < self.min_tokens:
                if current_doc is None:
                    current_doc = doc
                else:
                    # Пробуем объединить с предыдущим
                    combined_content = current_doc.page_content + "\n\n" + doc.page_content
                    combined_tokens = self._count_tokens(combined_content)
                    
                    # Если объединенный чанк не превышает максимум
                    if combined_tokens <= self.max_tokens:
                        # Объединяем
                        combined_metadata = current_doc.metadata.copy()
                        combined_metadata.update({
                            'token_count': combined_tokens,
                            'char_count': len(combined_content),
                            'char_token_ratio': len(combined_content) / combined_tokens if combined_tokens > 0 else 0,
                            'in_target_range': self.min_tokens <= combined_tokens <= self.max_tokens,
                            'merged_from_chunks': f"{current_doc.metadata.get('chunk_number', '?')},{doc.metadata.get('chunk_number', '?')}"
                        })
                        
                        current_doc = Document(
                            page_content=combined_content,
                            metadata=combined_metadata
                        )
                    else:
                        # Нельзя объединить - добавляем текущий и начинаем новый
                        optimized_docs.append(current_doc)
                        current_doc = doc
            else:
                # Чанк нормального размера
                if current_doc is not None:
                    optimized_docs.append(current_doc)
                    current_doc = None
                optimized_docs.append(doc)
        
        # Добавляем последний чанк
        if current_doc is not None:
            optimized_docs.append(current_doc)
        
        return optimized_docs

    def split(self, metadata: dict, text_to_split: str) -> List[Document]:
        strategy = self._analyze_document_structure(text_to_split)
        
        if strategy == "markdown":
            chunks = self._chunk_with_markdown_splitter(text_to_split, metadata)
        else:
            chunks = self._chunk_with_recursive_splitter(text_to_split, metadata)
        
        if not chunks or len(chunks) < 2:
            chunks = self._adaptive_chunking(text_to_split, metadata)
        
        return chunks
    
    def _analyze_document_structure(self, text: str) -> str:
        headers = len(re.findall(r'^#{1,4}\s+', text, re.MULTILINE))
        lists = len(re.findall(r'^\d+\.\s+|^[-*]\s+', text, re.MULTILINE))
        
        if headers >= 3 or (headers >= 1 and lists >= 2):
            return "markdown"
        else:
            return "recursive"
    
    def _chunk_with_markdown_splitter(self, text: str, base_metadata: dict) -> List[Document]:
        try:
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split_on,
                strip_headers=False
            )
            
            header_splits = markdown_splitter.split_text(text)
            final_chunks = []
            
            for doc in header_splits:
                content = doc.page_content
                if self._count_tokens(content) <= self.max_tokens:
                    final_chunks.append(content)
                else:
                    sub_chunks = self.recursive_splitter.split_text(content)
                    final_chunks.extend(sub_chunks)
            
            return self._convert_to_documents(final_chunks, base_metadata)
            
        except Exception:
            return self._chunk_with_recursive_splitter(text, base_metadata)
    
    def _chunk_with_recursive_splitter(self, text: str, base_metadata: dict) -> List[Document]:
        try:
            chunks = self.recursive_splitter.split_text(text)
            return self._convert_to_documents(chunks, base_metadata)
        except Exception:
            return self._adaptive_chunking(text, base_metadata)
    
    def _adaptive_chunking(self, text: str, base_metadata: dict) -> List[Document]:
        if self._count_tokens(text) < 1000:
            adaptive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name='cl100k_base',
                chunk_size=600,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""],
            )
        else:
            adaptive_splitter = self.recursive_splitter
        
        chunks = adaptive_splitter.split_text(text)
        return self._convert_to_documents(chunks, base_metadata)
    
    def _convert_to_documents(self, chunks: list, base_metadata: dict) -> List[Document]:
        documents = []
        
        for i, chunk_text in enumerate(chunks, 1):
            if self._count_tokens(chunk_text.strip()) < 100:
                continue
            
            char_count = len(chunk_text)
            token_count = self._count_tokens(chunk_text)
            ratio = char_count / token_count if token_count > 0 else 0
            
            russian_chars = len(re.findall(r'[А-Яа-яё]', chunk_text))
            russian_pct = (russian_chars / char_count) * 100 if char_count > 0 else 0
            artifacts = len(re.findall(r'#{1,6}|\*\*|\(\)|\[\]', chunk_text))
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_number": i,
                "char_count": char_count,
                "token_count": token_count,
                "char_token_ratio": ratio,
                "processing_date": datetime.now().isoformat(),
                "in_target_range": self.min_tokens <= token_count <= self.max_tokens,
                "ratio_in_target": 1.8 <= ratio <= 4.0,
                "russian_percentage": russian_pct,
                "artifact_count": artifacts,
                "high_quality": russian_pct >= 70 and artifacts <= 2,
                "splitter_type": "langchain_legal_tokens",
                "size_in_tokens": str(token_count)
            })
            
            document = Document(
                page_content=chunk_text.strip(),
                metadata=chunk_metadata
            )
            documents.append(document)
        
        # ВАЖНО: Объединяем мелкие чанки после создания документов
        documents = self._merge_small_chunks(documents)
        
        return documents
