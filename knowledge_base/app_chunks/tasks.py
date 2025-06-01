from celery import shared_task
from .models import Chunk
from utils.chunking import generate_chunks

@shared_task
def generate_chunks_task(source_type, source_id):
    """Задача для создания чанков."""
    from sources.models import URL, Document
    source = URL.objects.get(id=source_id) if source_type == 'url' else Document.objects.get(id=source_id)
    chunks = generate_chunks(source)
    return [chunk.id for chunk in chunks]