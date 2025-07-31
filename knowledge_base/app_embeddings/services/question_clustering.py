import os
import django
import re
from typing import List, Dict, Tuple
from collections import defaultdict

import nltk
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
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


class QuestionClusterer:
    """
    Класс для кластеризации пользовательских вопросов с помощью
    эмбеддингов и алгоритма HDBSCAN.
    Поддерживает сохранение и загрузку FAISS индекса, автогенерацию тегов,
    экспорт данных для визуализации.
    """

    def __init__(self, faiss_dir: str, model_name: str = "ai-forever/FRIDA"):
        """
        :param faiss_dir: Путь для хранения FAISS индекса.
        :param model_name: Название модели для эмбеддингов.
        """
        self.faiss_dir = faiss_dir
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        # self.clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
        self.clusterer = MiniBatchKMeans(n_clusters=10, batch_size=256, random_state=42)

        if not os.path.exists(self.faiss_dir):
            os.makedirs(self.faiss_dir)

        try:
            self.db = FAISS.load_local(self.faiss_dir, self.embeddings, index_name="index",
                                       allow_dangerous_deserialization=True)
            print("Загружена существующая")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.db = FAISS.from_documents([Document(page_content='', metadata={})], self.embeddings)
            print("Создана новая")


    def clean_text(self, text: str) -> str:
        """Приводит текст к нижнему регистру и убирает знаки препинания."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    def add_questions(self, raw_questions: List[Tuple[int, str]]):
        """
        Добавляет вопросы в FAISS индекс.

        :param raw_questions: Список кортежей (id, вопрос)
        """
        # Убираем дубликаты и пустые строки
        unique_q = {(qid, self.clean_text(q)) for qid, q in raw_questions if q.strip()}
        docs = [Document(page_content=q, metadata={"id": qid}) for qid, q in unique_q]
        self.db.add_documents(docs)
        self.db.save_local(folder_path=self.faiss_dir, index_name="index")

    def cluster_questions(self) -> Dict[int, List[Document]]:
        """
        Кластеризует все вопросы из FAISS.

        :return: Словарь {кластер_id: список Document}
        """
        docs = list(self.db.docstore._dict.values())

        if len(docs) < 5:
            return {-1: docs}

        questions = [doc.page_content for doc in docs]
        vectors = normalize(np.array(self.embeddings.embed_documents(questions)))

        labels = self.clusterer.fit_predict(vectors)

        clustered = defaultdict(list)
        for label, doc in zip(labels, docs):
            clustered[label].append(doc)

        return dict(clustered)

        return dict(clustered)

    def generate_tags(self, docs: List[Document], top_n: int = 5) -> List[str]:
        """
        Генерирует теги (ключевые слова) для кластера с помощью TF-IDF.

        :param docs: Список документов кластера.
        :param top_n: Кол-во топ-слов.
        :return: Список тегов.
        """
        texts = [doc.page_content for doc in docs]
        vectorizer = TfidfVectorizer(stop_words=russian_stopwords, max_features=1000)
        # vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        feature_array = np.array(vectorizer.get_feature_names_out())

        tfidf_mean = np.asarray(X.mean(axis=0)).ravel()
        top_indices = tfidf_mean.argsort()[::-1][:top_n]
        return feature_array[top_indices].tolist()

    def export_json(self, clusters: Dict[int, List[Document]]) -> str:
        """
        Экспортирует кластеры в JSON для визуализации.

        :param clusters: Результаты кластеризации.
        :return: JSON-строка с данными.
        """
        import json
        import umap

        all_docs = [doc for docs in clusters.values() for doc in docs]
        questions = [doc.page_content for doc in all_docs]
        vectors = self.embeddings.embed_documents(questions)
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(vectors)

        # id кластера для каждого документа
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

        :param clusters: Словарь кластеров.
        :param max_examples: Сколько вопросов показывать из каждого кластера.
        """
        sorted_items = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        for cluster_id, docs in sorted_items:
            if cluster_id == -1:
                print(f"\n=== 📎 Шум (не попали в кластеры) ({len(docs)} вопросов) ===")
            else:
                tags = self.generate_tags(docs)
                print(f"\n=== 🧠 Кластер {cluster_id} ({len(docs)} вопросов), теги: {', '.join(tags)} ===")
            for doc in docs[:max_examples]:
                print("  •", doc.page_content)


if __name__ == "__main__":
    FAISS_DIR = os.path.join(BASE_DIR, "media", "user_questions_faiss")
    print(f"{FAISS_DIR=}")

    # Берём id и текст вопросов из БД
    user_questions = ChatMessage.objects.filter(is_user=True).values_list('id', 'text')

    qc = QuestionClusterer(faiss_dir=FAISS_DIR)
    qc.add_questions(user_questions)
    clusters = qc.cluster_questions()
    qc.print_clusters(clusters)
