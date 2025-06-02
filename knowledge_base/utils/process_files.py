import hashlib


def compute_sha512(filepath):
    """
    Вычисляет SHA-512 для данного файла.
    Читает файл по частям, что позволяет обрабатывать большие файлы.
    """
    sha512 = hashlib.sha512()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha512.update(chunk)
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
        return None
    return sha512.hexdigest()
