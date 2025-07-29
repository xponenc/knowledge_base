from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from langchain.chat_models import ChatOpenAI

from app_api.models import ApiClient
from app_embeddings.services.multi_chain_factory import build_multi_chain
from fast_api.auth import get_api_client

multi_chain_router = APIRouter()

class ChainRequest(BaseModel):
    # kb_id: int
    query: str
    system_prompt: Optional[str] = None
    model: str = "gpt-4o-mini"

class SourceDocument(BaseModel):
    metadata: dict
    content: str

class ChainResponse(BaseModel):
    result: str
    source_documents: List[SourceDocument]

@multi_chain_router.post("/invoke", response_model=ChainResponse)
def invoke_chain(request: ChainRequest, client: ApiClient = Depends(get_api_client)):
    kb = client.knowledge_base
    model_name = request.model or kb.llm
    llm = ChatOpenAI(model=model_name, temperature=0)
    system_prompt = request.system_prompt or kb.system_instruction

    chain = build_multi_chain(
        kb_id=kb.pk,
        llm=llm,
    )

    result = chain.invoke({
        "input": request.query,
        "system_prompt": system_prompt,
    })

    docs = result.get("source_documents", [])
    parsed_docs = [
        SourceDocument(
            metadata=doc.metadata,
            content=doc.page_content
        ) for doc in docs
    ]

    return ChainResponse(result=result["result"], source_documents=parsed_docs)
