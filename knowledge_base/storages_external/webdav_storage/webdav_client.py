# storages_external/webdav_storage/webdav_client.py
import os
import tempfile

# import aiohttp
import requests
from urllib.parse import urljoin, unquote, urlparse
import xml.etree.ElementTree as ET

from django.core.signing import Signer, BadSignature

from knowledge_base.settings import TEMP_DIR
from storages_external.webdav_storage.base import BaseNetworkStorage
from utils.setup_logger import setup_logger

logger = setup_logger(__name__, log_dir="logs/documents_parsing", log_file="webdav_client.log")


class WebDavStorage(BaseNetworkStorage):
    """Класс для работы с WebDAV-хранилищем."""

    def __init__(self, credentials, check_connection=False):
        """
        Инициализация WebDAV-клиента.

        Args:
            credentials (dict): Учетные данные {'url': '...', 'root_path': '...', 'auth_type': 'token|basic', ...}

        Raises:
            ValueError: Если credentials некорректны или сервер недоступен.

            {
            "url": "https://cloud.academydpo.org/public.php/webdav/",
            "root_path": "documents/",
            "auth_type": "token",
            "token": "rqJWt7LzPGKcyNw"
            }

        {
            "url": "https://cloud.academydpo.org/public.php/webdav/",
            "root_path": "documents/",
            "auth_type": "basic",
            "username": "user",
            "password": "pass123"
        }
        """
        self.credentials = credentials
        self.validate_credentials()

        self.base_url = credentials['url']
        self.root_path = credentials['root_path']

        signer = Signer()

        # Установка auth в зависимости от auth_type
        auth_type = credentials.get('auth_type')
        if auth_type == 'token':
            if 'token' not in credentials:
                raise ValueError("Для auth_type='token' требуется ключ 'token'")
            try:
                token = signer.unsign(credentials['token'])
            except BadSignature:
                raise ValueError("Неверная подпись токена")
            self.auth = (token, "")
        elif auth_type == 'basic':
            if 'username' not in credentials or 'password' not in credentials:
                raise ValueError("Для auth_type='basic' требуются ключи 'username' и 'password'")
            try:
                password = signer.unsign(credentials['password'])
            except BadSignature:
                raise ValueError("Неверная подпись пароля")
            self.auth = (credentials['username'], password)
        else:
            raise ValueError("auth_type должен быть 'token' или 'basic'")

        if check_connection:
            self.check_connection()

    def validate_credentials(self):
        """Проверяет credentials и доступность сервера."""
        if not self.credentials or not isinstance(self.credentials, dict):
            raise ValueError(f"credentials не задан или не является словарем")
        required_keys = ['url', 'root_path', 'auth_type']
        missing_keys = [key for key in required_keys if key not in self.credentials]
        if missing_keys:
            raise ValueError(f"Отсутствуют ключи: {', '.join(missing_keys)}")
        if not self.credentials['url'].startswith('http'):
            raise ValueError("URL должен начинаться с http:// или https://")
        if self.credentials['auth_type'] not in ['token', 'basic']:
            raise ValueError("auth_type должен быть 'token' или 'basic'")
        if self.credentials['auth_type'] == 'token' and 'token' not in self.credentials:
            raise ValueError("Для auth_type='token' требуется ключ 'token'")
        if self.credentials['auth_type'] == 'basic' and (
                'username' not in self.credentials or 'password' not in self.credentials):
            raise ValueError("Для auth_type='basic' требуются ключи 'username' и 'password'")

    def check_connection(self):
        try:
            url = urljoin(self.credentials['url'], self.root_path)
            headers = {"Depth": "0", "Content-Type": "application/xml"}
            xml = """<?xml version="1.0" encoding="utf-8" ?>
                               <d:propfind xmlns:d="DAV:">
                                 <d:prop><d:resourcetype/></d:prop>
                               </d:propfind>
                       """
            response = requests.request("PROPFIND", url, data=xml, headers=headers, auth=self.auth)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка подключения к WebDAV: {e}")
            raise ValueError(f"Ошибка подключения: {e}")

    def list_directory(self, path):
        """
         Выполняет PROPFIND-запрос к указанной директории на WebDAV-сервере и возвращает
         список файлов в этой директории с их метаданными.

         Args:
             path (str): Относительный путь к директории на сервере.

         Returns:
             List[dict]: Список словарей с информацией о каждом найденном файле.
         """
        url = urljoin(self.base_url, path.lstrip('/'))
        headers = {"Depth": "1", "Content-Type": "application/xml"}
        xml = """<?xml version="1.0" encoding="utf-8" ?>
                <d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
                  <d:prop>
                    <d:getlastmodified/>
                    <d:getcontentlength/>
                    <d:resourcetype/>
                    <d:getetag/>
                  </d:prop>
                </d:propfind>
        """
        logger.info(f"PROPFIND запрос к {url}")
        try:
            response = requests.request("PROPFIND", url, data=xml, headers=headers, auth=self.auth)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Ошибка PROPFIND для {url}: {e}")

        ns = {'d': 'DAV:', 'oc': 'http://owncloud.org/ns'}
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as e:
            logger.error(f"Ошибка парсинга XML: {e}")
            raise

        items = []
        base_url = urljoin(self.base_url, path)
        for resp in root.findall('d:response', ns):
            href_elem = resp.find('d:href', ns)
            if href_elem is None:
                continue

            href = unquote(href_elem.text)
            if href.rstrip('/') == base_url.rstrip('/'):
                continue

            rel_path = href.lstrip('/')
            if rel_path.startswith('public.php/webdav/'):
                rel_path = rel_path[len('public.php/webdav/'):]

            parsed = urlparse(href)
            file_name = parsed.path.rstrip('/').split('/')[-1]

            resourcetype = resp.find('d:propstat/d:prop/d:resourcetype', ns)
            is_dir = resourcetype is not None and resourcetype.find('d:collection', ns) is not None
            if not is_dir:
                item = {
                    'url': urljoin(self.base_url, rel_path.lstrip('/')),
                    'path': rel_path,
                    'file_name': file_name,
                    'is_dir': is_dir,
                    'last_modified': resp.findtext('d:propstat/d:prop/d:getlastmodified', default=None, namespaces=ns),
                    'size': int(resp.findtext('d:propstat/d:prop/d:getcontentlength', default="0",
                                              namespaces=ns)) if not is_dir else 0,
                    'file_id': resp.findtext('d:propstat/d:prop/d:getetag', default=None, namespaces=ns),
                }
                items.append(item)
        return items

    def propfind(self, path):
        """
        Выполняет прямой PROPFIND-запрос к WebDAV-серверу и возвращает XML-ответ как строку.

        Args:
            path (str): Относительный путь к файлу или директории.

        Returns:
            str: XML-ответ от сервера.
        """
        url = urljoin(self.base_url, path)
        headers = {"Depth": "1", "Content-Type": "application/xml"}
        xml = """<?xml version="1.0" encoding="utf-8" ?>
                    <d:propfind xmlns:d="DAV:">
                      <d:prop>
                        <d:getlastmodified/>
                        <d:getcontentlength/>
                        <d:resourcetype/>
                        <d:getetag/>
                      </d:prop>
                    </d:propfind>
        """
        logger.info(f"PROPFIND запрос к {url}")
        try:
            response = requests.request("PROPFIND", url, data=xml, headers=headers, auth=self.auth)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Ошибка PROPFIND для {url}: {e}")
            raise
        return response.text

    def parse_propfind(self, xml_response, base_url):
        """
        Парсит XML-ответ от WebDAV-сервера, извлекая информацию о файлах и папках.

        Args:
            xml_response (str): XML-ответ от `propfind`.
            base_url (str): Абсолютный базовый URL (для фильтрации самих себя).

        Returns:
            List[dict]: Список файлов и директорий с метаданными.
        """
        ns = {'d': 'DAV:'}
        root = ET.fromstring(xml_response)
        items = []

        for resp in root.findall('d:response', ns):
            href_elem = resp.find('d:href', ns)
            if href_elem is None:
                continue

            href = unquote(href_elem.text)
            if href.rstrip('/') == base_url.rstrip('/'):
                continue

            if href.startswith('/'):
                rel_path = href.lstrip('/')
            else:
                rel_path = href
            if rel_path.startswith('public.php/webdav/'):
                rel_path = rel_path[len('public.php/webdav/'):]

            parsed = urlparse(href)
            file_name = parsed.path.rstrip('/').split('/')[-1]

            resourcetype = resp.find('d:propstat/d:prop/d:resourcetype', ns)
            is_dir = resourcetype is not None and resourcetype.find('d:collection', ns) is not None

            item = {
                'path': rel_path,
                'file_name': file_name,
                'is_dir': is_dir,
                'last_modified': resp.findtext('d:propstat/d:prop/d:getlastmodified', default=None, namespaces=ns),
                'size': int(
                    resp.findtext('d:propstat/d:prop/d:getcontentlength', default="0",
                                  namespaces=ns)) if not is_dir else 0,
                'file_id': resp.findtext('d:propstat/d:prop/d:getetag', default=None, namespaces=ns),
                'url': urljoin(self.base_url, rel_path.lstrip('/'))
            }

            if is_dir:
                logger.info(f"Найдена папка: {file_name}")
            else:
                logger.info(
                    f"Найден файл: {file_name}, размер: {item['size']} байт, изменен {item['last_modified']}, file_id: {item['file_id']}")

            items.append(item)
        return items

    def list_files_recursive(self, path):
        """
           Рекурсивно обходит директорию на WebDAV и возвращает список всех файлов с метаданными.

           Args:
               path (str): Относительный путь к стартовой директории.

           Returns:
               List[dict]: Список всех найденных файлов (включая вложенные).
           """
        xml_response = self.propfind(path)
        base_url = urljoin(self.base_url, path)
        items = self.parse_propfind(xml_response, base_url)

        all_files = []
        for item in items:
            if item['is_dir']:
                sub_path = item['path']
                if sub_path.rstrip('/') == path.rstrip('/'):
                    continue
                logger.info(f"Рекурсивный вызов для поддиректории: {sub_path}")
                all_files.extend(self.list_files_recursive(sub_path))
            else:
                all_files.append(item)
        return all_files

    def download_file_to_disk_sync(self, file_url):
        response = requests.get(file_url, stream=True, auth=self.auth)
        response.raise_for_status()

        temp_dir = TEMP_DIR
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        original_filename = os.path.basename(file_url)

        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        temp_path = tmp_file.name

        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)

        return temp_path, original_filename
