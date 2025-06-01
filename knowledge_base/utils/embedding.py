from langchain.text_splitter import RecursiveCharacterTextSplitter
from chunks.models import Chunk

def generate_chunks(source):
    """
    Создает чанки из контента источника.
    :param source: Объект URL или Document.
    :return: Список объектов Chunk.
    """
    source_type = 'url' if hasattr(source, 'url') else 'document'
    source_id = source.id
    content = source.cleaned_content
    language = source.language
    chunk_size = 1000  # Адаптивно
    chunk_overlap = 200  # Адаптивно
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(content)
    chunk_objects = []
    for chunk_text in chunks:
        chunk = Chunk.objects.create(
            source_type=source_type,
            source_id=source_id,
            content=chunk_text,
            language=language,
            chunking_model='recursive',
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunk_objects.append(chunk)
    return chunk_objects