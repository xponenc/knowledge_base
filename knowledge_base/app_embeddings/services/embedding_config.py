import os

from knowledge_base.settings import BASE_DIR
from dotenv import load_dotenv

load_dotenv()


MODELS = {
    "frida": {
        "name": "ai-forever/FRIDA",
        "path": os.path.join(BASE_DIR, "app_embeddings", "embedding_store", "frida_faiss_index_db")
    },
    "sbert_ru": {
        "name": "ai-forever/sbert_large_nlu_ru",
        "path": BASE_DIR / "faiss_db/sbert_faiss_index"
    },
    "e5": {
        "name": "intfloat/multilingual-e5-base",
        "path": BASE_DIR / "faiss_db/e5_faiss_index"
    },
    "oai": {
        "name": "openai",
        "path": BASE_DIR / "faiss_db/oai_faiss_index"
    },
}

RERANKER_MODEL = "DiTy/cross-encoder-russian-msmarco"
# RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
FAISS_THRESHOLD = 0.8
TOP_N = 5

