import os

import django
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

from app_embeddings.services.multi_chain_factory import get_retriever_only

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "knowledge_base.settings")
django.setup()

from fast_api.auth import get_api_client
from app_api.models import ApiClient
from app_embeddings.services.ensemble_chain_factory import get_cached_ensemble_retriever
from app_embeddings.services.retrieval_engine import append_links_to_documents, append_url_source_to_documents

multi_retriever_router = APIRouter()


class RetrieverRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос пользователя")
    top_k: Optional[int] = Field(5, description="Количество документов для возврата (по умолчанию 5)")


class SourceDocument(BaseModel):
    metadata: dict
    content: str


class RetrieverResponse(BaseModel):
    documents: List[SourceDocument]


@multi_retriever_router.post("/search", response_model=RetrieverResponse)
def search_with_retriever(
    request: RetrieverRequest,
    client: ApiClient = Depends(get_api_client)
):
    """
    Поиск документов с помощью ensemble retriever.
    """
    kb = client.knowledge_base
    retriever = get_retriever_only(kb_id=kb.id)

    results = retriever.get_relevant_documents(request.query)

    # Ограничиваем top_k # TODO вообще оно ограничватеся в самом ретривере, но не доделан механизм передачи настраиваемого параметра
    if request.top_k and request.top_k > 0:
        results = results[: request.top_k]

    enriched_documents = append_links_to_documents(
        results)  # Обогащение документов ссылками на полезные источники из метаданных
    enriched_documents = append_url_source_to_documents(
        enriched_documents)  # Обогащение документов ссылками на источник

    # Приводим к pydantic-модели
    parsed_docs = [
        SourceDocument(
            metadata=doc.metadata,
            content=doc.page_content
        )
        for doc in enriched_documents
    ]

    return RetrieverResponse(documents=parsed_docs)



