import os
import time

import django
import re
from typing import List, Dict, Tuple, Any
from collections import defaultdict

import nltk
import numpy as np
from faiss import FileIOReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import hdbscan
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

# Django init
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "knowledge_base.settings")
django.setup()

from app_chat.models import ChatMessage
from knowledge_base.settings import BASE_DIR

nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")


import logging
from typing import List, Dict
from collections import defaultdict
import time
import hashlib

import django
import re
import nltk
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Django init
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "knowledge_base.settings")
django.setup()

from app_chat.models import ChatMessage
from knowledge_base.settings import BASE_DIR

nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")

class QuestionClusterer:
    # Статический словарь для кэша: {kb_pk: {text: embedding}}
    embedding_cache = {}
    result_cache = {}

    def __init__(self, kb_pk: int, model_name: str = "ai-forever/FRIDA"):
        self.kb_pk = kb_pk
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.faiss_dir = os.path.join(BASE_DIR, "media", "kb", str(kb_pk), "user_questions_faiss")
        self.clusterer = MiniBatchKMeans(n_clusters=10, batch_size=256, random_state=42)

        # Инициализируем кэш для данной базы знаний, если он ещё не существует
        if self.kb_pk not in self.embedding_cache:
            self.embedding_cache[self.kb_pk] = {}

        if not os.path.exists(self.faiss_dir):
            os.makedirs(self.faiss_dir)

        try:
            self.db = FAISS.load_local(self.faiss_dir, self.embeddings, index_name="index",
                                       allow_dangerous_deserialization=True)
            logger.info(f"Загружена существующая база кластеров для kb_pk={self.kb_pk}")
        except RuntimeError as e:
            if "could not open" in str(e):
                self.db = None
                logger.info(f"Создана новая база кластеров для kb_pk={self.kb_pk}")
            else:
                raise

    @classmethod
    def clear_cache(cls, kb_pk: int):
        if kb_pk in cls.embedding_cache:
            del cls.embedding_cache[kb_pk]
            logger.info(f"Кэш embedding_cache для kb_pk={kb_pk} очищен")
        if kb_pk in cls.result_cache:
            del cls.result_cache[kb_pk]
            logger.info(f"Кэш result_cache для kb_pk={kb_pk} очищен")

    def embed_with_cache(self, texts: List[str]) -> List[List[float]]:
        """
        Получает эмбеддинги для списка текстов, используя кэш, уникальный для kb_pk.
        """
        start_time = time.monotonic()
        cleaned_texts = [self.clean_text(text) for text in texts]
        result = []
        cache_hits = 0
        total_texts = len(texts)

        texts_to_embed = []
        indices_to_embed = []
        for idx, text in enumerate(cleaned_texts):
            if text in self.embedding_cache[self.kb_pk]:
                result.append(self.embedding_cache[self.kb_pk][text])
                cache_hits += 1
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(idx)
                result.append(None)

        if texts_to_embed:
            new_embeddings = self.embeddings.embed_documents(texts_to_embed)
            for text, embedding, idx in zip(texts_to_embed, new_embeddings, indices_to_embed):
                self.embedding_cache[self.kb_pk][text] = embedding
                result[idx] = embedding

        logger.info(f"Кэш для kb_pk={self.kb_pk}: {cache_hits}/{total_texts} текстов найдено в кэше "
                    f"({cache_hits/total_texts*100:.2f}%)")
        logger.info(f"Время выполнения embed_with_cache: {time.monotonic() - start_time:.2f} секунд")
        return result

    def clean_text(self, text: str) -> str:
        """Приводит текст к нижнему регистру и убирает знаки препинания."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    def add_questions(self, raw_questions: List[Tuple[int, str]]):
        """
        Добавляет вопросы в FAISS индекс.
        """
        unique_q = {(qid, self.clean_text(q)) for qid, q in raw_questions if q.strip()}
        docs = [Document(page_content=q, metadata={"id": qid}) for qid, q in unique_q]
        if self.db is None:
            self.db = FAISS.from_documents(docs, self.embeddings)
        else:
            self.db.add_documents(docs)
        self.db.save_local(folder_path=self.faiss_dir, index_name="index")

    def cluster_questions(self) -> Dict[int, Dict[str, Any]]:
        """
        Кластеризует все вопросы из FAISS и сразу генерирует теги для каждого кластера.
        Возвращает словарь:
        {
          cluster_id: {
            "docs": [Document, ...],
            "tags": [str, ...]
          },
          ...
        }
        """

        if self.kb_pk in QuestionClusterer.result_cache:
            logger.info(f"cluster_questions: взято из кэша для kb_pk={self.kb_pk}")
            # Возвращаем только структуры с тегами и доками
            cached_clusters = QuestionClusterer.result_cache[self.kb_pk][0]
            return cached_clusters

        start_time = time.monotonic()
        docs = list(self.db.docstore._dict.values())

        if len(docs) < 5:
            clusters = {-1: docs}
        else:
            questions = [doc.page_content for doc in docs]
            vectors = normalize(np.array(self.embed_with_cache(questions)))
            labels = self.clusterer.fit_predict(vectors)

            clustered = defaultdict(list)
            for label, doc in zip(labels, docs):
                clustered[label].append(doc)

            clusters = dict(clustered)

        # Генерация тегов для каждого кластера
        clusters_with_tags = {}
        for cluster_id, docs_in_cluster in clusters.items():
            tags = self.generate_tags(docs_in_cluster, top_n=5)
            clusters_with_tags[cluster_id] = {
                "docs": docs_in_cluster,
                "tags": tags,
            }

        # Экспорт JSON для визуализации
        json_data = self.export_json(clusters)

        # Сохраняем в кэш:
        # в кэше теперь кластеры с тегами + json
        QuestionClusterer.result_cache[self.kb_pk] = (clusters_with_tags, json_data)

        logger.info(f"Кластеры с тегами для kb_pk={self.kb_pk} сохранены в кэш")
        logger.info(f"Время выполнения cluster_questions: {time.monotonic() - start_time:.2f} сек")

        return clusters_with_tags

    def generate_tags(self, docs: List[Document], top_n: int = 5) -> List[str]:
        """
        Генерирует теги (ключевые слова) для кластера с помощью TF-IDF.
        """
        print(f"generate_tags {docs=}")
        texts = [doc.page_content for doc in docs]
        vectorizer = TfidfVectorizer(
            stop_words=russian_stopwords,
            max_features=1000,
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        try:
            X = vectorizer.fit_transform(texts)
            if X.shape[1] == 0:
                return ["(недостаточно данных)"]
            feature_array = np.array(vectorizer.get_feature_names_out())
            tfidf_mean = np.asarray(X.mean(axis=0)).ravel()
            top_indices = tfidf_mean.argsort()[::-1][:top_n]
            return feature_array[top_indices].tolist()
        except ValueError:
            return ["(не удалось выделить теги)"]

    def export_json(self, clusters: Dict[int, List[Document]]) -> str:
        """
        Экспортирует кластеры в JSON для визуализации.
        """
        import json
        import umap

        all_docs = [doc for docs in clusters.values() for doc in docs]
        questions = [doc.page_content for doc in all_docs]
        vectors = self.embed_with_cache(questions)
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(vectors)

        cluster_map = {}
        idx = 0
        for cluster_id, docs in clusters.items():
            for _ in docs:
                cluster_map[idx] = cluster_id
                idx += 1

        data = []
        for i, doc in enumerate(all_docs):
            data.append({
                "id": doc.metadata.get("id"),
                "text": doc.page_content,
                "cluster": int(cluster_map[i]),
                "x": float(embedding_2d[i, 0]),
                "y": float(embedding_2d[i, 1]),
            })

        return json.dumps(data, ensure_ascii=False)

    def print_clusters(self, clusters: Dict[int, List[Document]], max_examples: int = 5):
        """
        Красивый вывод кластеров в консоль.
        """
        sorted_items = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        for cluster_id, data in sorted_items:
            print(data)
            docs = data.get("docs")
            if cluster_id == -1:
                print(f"\n=== 📎 Шум (не попали в кластеры) ({len()} вопросов) ===")
            else:
                print(f"\n=== 🧠 Кластер {cluster_id} ({len(docs)} вопросов), теги: {', '.join( data.get('tags'))} ===")
            for doc in docs[:max_examples]:
                print("  •", doc.page_content)


if __name__ == "__main__":
    KB_PK = 1
    start_time = time.monotonic()
    # Берём id и текст вопросов из БД
    user_questions = ChatMessage.objects.filter(is_user=True).values_list('id', 'text')

    qc = QuestionClusterer(kb_pk=KB_PK)
    qc.add_questions(user_questions)
    clusters = qc.cluster_questions()
    qc.print_clusters(clusters)

    print("Длительность: ", time.monotonic() - start_time)
