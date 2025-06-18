import faiss
import pickle
import numpy as np
from collections import defaultdict

# Путь к файлам
faiss_index_path = "./frida_faiss_index_db/index.faiss"
pkl_metadata_path = "./frida_faiss_index_db/index.pkl"

# Загружаем метаданные
with open(pkl_metadata_path, "rb") as f:
    metadata = pickle.load(f)

print("Тип объекта из index.pkl:", type(metadata))

# Проверяем, что metadata — это кортеж
if isinstance(metadata, tuple):
    print("Длина кортежа:", len(metadata))
    docstore, id_map = metadata  # Распаковываем кортеж
    print("Тип docstore:", type(docstore))
    print("Тип id_map:", type(id_map))
else:
    raise RuntimeError("Ожидался кортеж в metadata, но получен другой тип")

# Извлечение документов из InMemoryDocstore
all_docs = []
if hasattr(docstore, '_dict'):
    for key, doc in docstore._dict.items():
        text = getattr(doc, 'page_content', None) or getattr(doc, 'text', None) or ""
        all_docs.append({'id': key, 'text': text})
else:
    raise RuntimeError("Не найден атрибут _dict в docstore")

# Создаем словарь id->данные
id_to_data = {}
for i, doc in enumerate(all_docs):
    id_to_data[i] = {'text': doc['text'], 'source': 'unknown'}

SIMILARITY_THRESHOLD = 0.99

print("Загружаем FAISS индекс...")
index = faiss.read_index(faiss_index_path)
total_vectors = index.ntotal
dimension = index.d
print(f"Всего векторов: {total_vectors}, размерность: {dimension}")


def find_duplicates(index, id_to_data, similarity_threshold=SIMILARITY_THRESHOLD, top_k=10):
    clusters = defaultdict(list)
    vector_id_to_cluster = {}
    cluster_counter = 0

    batch_size = 1000
    total = index.ntotal

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_size_actual = end - start

        vectors = np.array([index.reconstruct(i) for i in range(start, end)])
        distances, neighbors = index.search(vectors, top_k)

        for j in range(batch_size_actual):
            vector_id = start + j
            if vector_id in vector_id_to_cluster:
                continue

            clusters[cluster_counter].append(vector_id)
            vector_id_to_cluster[vector_id] = cluster_counter

            for dist, neighbor_id in zip(distances[j], neighbors[j]):
                if neighbor_id == vector_id:
                    continue
                if dist >= similarity_threshold and neighbor_id not in vector_id_to_cluster:
                    clusters[cluster_counter].append(neighbor_id)
                    vector_id_to_cluster[neighbor_id] = cluster_counter

            cluster_counter += 1

        print(f"Обработано векторов: {end}/{total}")

    return clusters


print("Ищем дубликаты...")
clusters = find_duplicates(index, id_to_data)


def print_duplicate_clusters(clusters, id_to_data, min_size=2):
    duplicate_clusters = {cid: vids for cid, vids in clusters.items() if len(vids) >= min_size}
    print(f"Найдено кластеров с дубликатами (size >= {min_size}): {len(duplicate_clusters)}")

    for cluster_id, vector_ids in duplicate_clusters.items():
        print(f"\nКластер {cluster_id} ({len(vector_ids)} векторов):")
        for vid in vector_ids:
            data = id_to_data.get(vid, {})
            text_preview = data.get('text', '')[:100].replace('\n', ' ')
            source = data.get('source', 'unknown')
            # print(f"  ID: {vid}, Источник: {source}, Текст: {text_preview}...")
            print(f"  ID: {vid}, Источник: {source}, Текст: {data.get('text', '')}")


print_duplicate_clusters(clusters, id_to_data)
