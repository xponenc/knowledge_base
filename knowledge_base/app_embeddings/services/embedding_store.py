from app_embeddings.services.embedding_config import MODELS, RERANKER_MODEL
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder
import torch

_selected_model = "frida"

def load_embedding(model_name: str):
    if model_name == "openai":
        return OpenAIEmbeddings()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8}
    )

def get_vectorstore():
    config = MODELS[_selected_model]
    embeddings = load_embedding(config["name"])
    return FAISS.load_local(
        str(config["path"]),
        embeddings=embeddings,
        index_name="index",
        allow_dangerous_deserialization=True
    )

def get_reranker():
    return CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


VECTORSTORE = get_vectorstore()
RERANKER = get_reranker()
