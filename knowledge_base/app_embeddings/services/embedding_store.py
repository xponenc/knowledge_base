import glob
import os
from pathlib import Path

from app_embeddings.services.embedding_config import MODELS, RERANKER_MODEL
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder
import torch

_selected_model = "frida"

# def load_embedding(model_name: str):
#     if model_name == "openai":
#         return OpenAIEmbeddings()
#     return HuggingFaceEmbeddings(
#         model_name=model_name,
#         encode_kwargs={"normalize_embeddings": True, "batch_size": 8}
#     )


def load_embedding(model_name: str):
    if model_name == "openai":
        return OpenAIEmbeddings()

    local_path = get_local_model_path(model_name)
    model_path = local_path if local_path else model_name  # если есть локальный путь — используем его

    return HuggingFaceEmbeddings(
        model_name=model_path,
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

# def get_reranker():
#     return CrossEncoder(
#         RERANKER_MODEL,
#         max_length=512,
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )


def get_reranker():
    local_path = get_local_model_path(RERANKER_MODEL)
    model_path = local_path if local_path else RERANKER_MODEL

    return CrossEncoder(
        model_path,
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


def get_local_model_path(repo_id: str):
    """Возвращает путь к локальной модели Hugging Face, если она загружена"""
    custom_base = Path(__file__).resolve().parent.parent / "local_models"
    candidates = list(custom_base.glob(f"**/{repo_id.replace('/', '-')}/"))  # sentence-transformers обычно кладёт сюда
    if candidates:
        return str(candidates[0])
    return None


VECTORSTORE = get_vectorstore()
RERANKER = get_reranker()
