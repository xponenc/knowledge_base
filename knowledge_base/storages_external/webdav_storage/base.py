from abc import ABC, abstractmethod


class BaseNetworkStorage(ABC):
    """
    Абстрактный класс для всех типов хранилищ.
    """

    @abstractmethod
    def list_directory(self, path: str) -> list:
        """
        Возвращает список файлов с параметрами начиная с path и далее рекурсивно по всем поддиректориям.
        :param path: путь к корневой директории
        :return: список из словарей вида: {
        }
        """
        pass

    @abstractmethod
    def download_file_to_disk_sync(self, file_url: str) -> tuple[str, str]:
        """
        Скачивает файл по URL и сохраняет его на диск.

        :param file_url: Полный URL файла
        :return: (путь до временного файла, оригинальное имя файла)
        """
        pass
