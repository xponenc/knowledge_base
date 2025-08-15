# Инициализация Django
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "knowledge_base.settings")
django.setup()

from app_embeddings.services.ensemble_chain_factory import get_cached_ensemble_retriever


retriever = get_cached_ensemble_retriever(kb_id=1)


# Создаем функцию для поиска по topic_phrases
def search_with_retriever(inputs):
    topic_phrases = inputs.get("topic_phrases", "").content
    if not topic_phrases:
        return {"search_index": None}
    search_results = retriever.invoke(topic_phrases)
    search_index = ""
    for index, document in enumerate(search_results, start=1):
        search_index += f"Документ {index}:\n{document.page_content}\n"
    return {"search_index": search_index}