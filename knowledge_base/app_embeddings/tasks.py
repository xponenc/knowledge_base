from celery import shared_task
from .models import Embedding
from utils.embedding import generate_embedding

@shared_task
def generate_embedding_task(chunk_id):
    """Задача для создания эмбеддинга."""
    from chunks.models import Chunk
    chunk = Chunk.objects.get(id=chunk_id)
    embedding = generate_embedding(chunk)
    return embedding.id