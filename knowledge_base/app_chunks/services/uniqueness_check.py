###
##
##  тестовый модуль для поиска идентичных чанков
##
###

import re
import sqlite3
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


def normalize_text(text):
    """Нормализует текст: нижний регистр, убирает пробелы и пунктуацию."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def cluster_similar_chunks_with_faiss(db_path, table_name, text_column, source_column, batch_size=1000,
                                      similarity_threshold=0.8, max_features=5000):
    # Создаём единый векторизатор
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Собираем все чанки для создания словаря
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, {text_column}, {source_column} FROM {table_name}")
        all_chunks = []
        all_ids = []
        all_sources = []

        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            all_chunks.extend([normalize_text(row[1]) for row in batch])
            all_ids.extend([row[0] for row in batch])
            all_sources.extend([row[2] for row in batch])

    # Преобразуем все чанки в TF-IDF векторы
    tfidf_matrix = vectorizer.fit_transform(all_chunks).toarray().astype(np.float32)

    # Создаём FAISS индекс
    dimension = tfidf_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Используем Inner Product для косинусного сходства
    faiss.normalize_L2(tfidf_matrix)  # Нормализуем векторы для косинусного сходства
    index.add(tfidf_matrix)

    # Кластеризация
    clusters = defaultdict(list)
    chunk_id_to_cluster = {}
    cluster_counter = 0

    # Ищем ближайших соседей
    D, I = index.search(tfidf_matrix, k=10)  # Ищем 10 ближайших соседей
    for i in range(len(all_chunks)):
        if i in chunk_id_to_cluster:
            continue
        clusters[cluster_counter].append((all_chunks[i], all_sources[i], all_ids[i]))
        chunk_id_to_cluster[i] = cluster_counter

        for dist, neighbor_idx in zip(D[i], I[i]):
            if neighbor_idx != i and dist > similarity_threshold and neighbor_idx not in chunk_id_to_cluster:
                clusters[cluster_counter].append(
                    (all_chunks[neighbor_idx], all_sources[neighbor_idx], all_ids[neighbor_idx]))
                chunk_id_to_cluster[neighbor_idx] = cluster_counter
        cluster_counter += 1

    # Сохранение результатов
    with open("chunk_clusters.txt", "w", encoding="utf-8") as f:
        for cluster_id, chunks in clusters.items():
            f.write(f"Кластер {cluster_id} ({len(chunks)} чанков):\n")
            for chunk, source, chunk_id in chunks:
                f.write(f"  ID: {chunk_id}, Источник: {source}, Текст: {chunk[:100]}...\n")
            f.write("\n")

    return clusters


# Пример использования
db_path = "RAG_DB/SQLite_DB/chunks.db"
table_name = "chunks"
text_column = "text"
source_column = "source"
clusters = cluster_similar_chunks_with_faiss(db_path, table_name, text_column, source_column)
print(f"Найдено {len(clusters)} кластеров похожих чанков")