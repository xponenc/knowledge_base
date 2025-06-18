###
##
##  тестовый модуль для поиска идентичных векторов
##
###
import re
import sqlite3
import faiss
import numpy as np
from collections import defaultdict
import logging

# Настройка логирования (как в вашем main.py)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/deduplicate_vectors.log")
    ]
)


def normalize_text(text):
    """Нормализует текст: нижний регистр, убирает пробелы и пунктуацию."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def find_duplicate_vectors(db_path, faiss_index_path, table_name, text_column, source_column,
                           batch_size=1000, similarity_threshold=0.99):
    """
    Находит дубликаты векторов в FAISS и группирует их в кластеры.
    Возвращает кластеры с чанками и источниками.
    """
    # Загружаем FAISS-индекс
    logging.info("Загрузка FAISS-индекса...")
    index = faiss.read_index(faiss_index_path)
    total_vectors = index.ntotal
    dimension = index.d
    logging.info(f"Найдено {total_vectors} векторов размерности {dimension}")

    # Проверяем, нормализован ли индекс для косинусного сходства
    if not isinstance(index, faiss.IndexFlatIP):
        logging.warning("Индекс не использует Inner Product. Переключаемся на IndexFlatIP.")
        new_index = faiss.IndexFlatIP(dimension)
        vectors = index.reconstruct_n(0, total_vectors)
        faiss.normalize_L2(vectors)
        new_index.add(vectors)
        index = new_index

    # Инициализация кластеров
    clusters = defaultdict(list)
    vector_id_to_cluster = {}
    cluster_counter = 0

    # Подключаемся к SQLite для извлечения метаданных
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Обрабатываем векторы батчами
        for i in range(0, total_vectors, batch_size):
            batch_size_actual = min(batch_size, total_vectors - i)
            batch_vectors = index.reconstruct_n(i, batch_size_actual)

            # Поиск ближайших соседей
            D, I = index.search(batch_vectors, k=10)  # Ищем 10 ближайших соседей
            logging.info(f"Обработан батч {i}–{i + batch_size_actual}")

            # Извлекаем метаданные для текущего батча
            batch_ids = list(range(i, i + batch_size_actual))
            cursor.execute(
                f"SELECT id, {text_column}, {source_column} FROM {table_name} WHERE id IN ({','.join(['?'] * len(batch_ids))})",
                batch_ids)
            chunk_data = {row[0]: (normalize_text(row[1]), row[2]) for row in cursor.fetchall()}

            # Кластеризация
            for j in range(batch_size_actual):
                vector_id = i + j
                if vector_id in vector_id_to_cluster:
                    continue

                # Добавляем текущий вектор в новый кластер
                chunk, source = chunk_data.get(vector_id, ("", ""))
                clusters[cluster_counter].append((vector_id, chunk, source))
                vector_id_to_cluster[vector_id] = cluster_counter

                # Проверяем соседей
                for dist, neighbor_id in zip(D[j], I[j]):
                    if neighbor_id != vector_id and dist > similarity_threshold and neighbor_id not in vector_id_to_cluster:
                        neighbor_chunk, neighbor_source = chunk_data.get(neighbor_id, ("", ""))
                        clusters[cluster_counter].append((neighbor_id, neighbor_chunk, neighbor_source))
                        vector_id_to_cluster[neighbor_id] = cluster_counter

                cluster_counter += 1

    # Сохранение результатов
    logging.info("Сохранение результатов в файл...")
    with open("vector_clusters.txt", "w", encoding="utf-8") as f:
        for cluster_id, vectors in clusters.items():
            if len(vectors) > 1:  # Сохраняем только кластеры с дубликатами
                f.write(f"Кластер {cluster_id} ({len(vectors)} векторов):\n")
                for vector_id, chunk, source in vectors:
                    f.write(f"  ID: {vector_id}, Источник: {source}, Текст: {chunk[:100]}...\n")
                f.write("\n")

    # Анализ источников
    source_counts = defaultdict(lambda: defaultdict(int))
    for cluster_id, vectors in clusters.items():
        if len(vectors) > 1:  # Анализируем только кластеры с дубликатами
            for _, _, source in vectors:
                source_counts[cluster_id][source] += 1

    with open("source_analysis.txt", "w", encoding="utf-8") as f:
        for cluster_id, sources in source_counts.items():
            f.write(f"Кластер {cluster_id}:\n")
            for source, count in sources.items():
                f.write(f"  Источник: {source}, Количество векторов: {count}\n")
            f.write("\n")

    logging.info(f"Найдено {len([c for c in clusters.values() if len(c) > 1])} кластеров с дубликатами")
    return clusters, source_counts


# Функция для вывода кластеров в консоль
def print_clusters(clusters, min_vectors=2):
    for cluster_id, vectors in clusters.items():
        if len(vectors) >= min_vectors:
            print(f"Кластер {cluster_id} ({len(vectors)} векторов):")
            for vector_id, chunk, source in vectors:
                print(f"  ID: {vector_id}, Источник: {source}, Текст: {chunk[:100]}...")
            print()


# Пример использования
db_path = "RAG_DB/SQLite_DB/chunks.db"
faiss_index_path = "RAG_DB/faiss_index"
table_name = "chunks"
text_column = "text"
source_column = "source"
clusters, source_counts = find_duplicate_vectors(db_path, faiss_index_path, table_name, text_column, source_column)
print(f"Найдено {len([c for c in clusters.values() if len(c) > 1])} кластеров с дубликатами")
print_clusters(clusters, min_vectors=2)