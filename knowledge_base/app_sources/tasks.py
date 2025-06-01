from celery import shared_task
from .models import URL, Document
from chunks.models import Chunk
from embeddings.models import Embedding
from utils.parsing import parse_url
from utils.cloud_storage import sync_document
from utils.chunking import generate_chunks
from utils.embedding import generate_embedding

@shared_task
def parse_url_task(url_id):
    """Задача для парсинга URL."""
    url = URL.objects.get(id=url_id)
    result = parse_url(url)
    if result.get('success'):
        url.update_status('parsed')
        chunks = generate_chunks(url)
        for chunk in chunks:
            embedding = generate_embedding(chunk)
    else:
        url.error_message = result.get('error')
        url.update_status('error')

@shared_task
def sync_document_task(document_id):
    """Задача для синхронизации документа."""
    document = Document.objects.get(id=document_id)
    result = sync_document(document)
    if result.get('success'):
        document.update_status('synced')
        chunks = generate_chunks(document)
        for chunk in chunks:
            embedding = generate_embedding(chunk)
    else:
        document.error_message = result.get('error')
        document.update_status('error')