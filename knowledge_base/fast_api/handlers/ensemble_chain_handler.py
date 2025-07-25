from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from langchain.chat_models import ChatOpenAI

from app_api.models import ApiClient
from app_embeddings.services.ensemble_chain_factory import build_ensemble_chain
from fast_api.auth import get_api_client

ensemble_chain_router = APIRouter()

class ChainRequest(BaseModel):
    kb_id: int
    query: str
    system_prompt: Optional[str] = None
    model: str = "gpt-4o-mini"

class SourceDocument(BaseModel):
    metadata: dict
    content: str

class ChainResponse(BaseModel):
    result: str
    source_documents: List[SourceDocument]

@ensemble_chain_router.post("/invoke", response_model=ChainResponse)
def invoke_chain(request: ChainRequest, client: ApiClient = Depends(get_api_client)):
    llm = ChatOpenAI(model=request.model, temperature=0)
    chain = build_ensemble_chain(request.kb_id, llm)

    result = chain.invoke({
        "input": request.query,
        "system_prompt": request.system_prompt,
    })

    docs = result.get("context", [])
    parsed_docs = [
        SourceDocument(
            metadata=doc.metadata,
            content=doc.page_content
        ) for doc in docs
    ]

    return ChainResponse(result=result["answer"], source_documents=parsed_docs)