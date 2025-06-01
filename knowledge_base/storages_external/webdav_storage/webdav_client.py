# storages_external/webdav_storage/webdav_client.py
import requests
from urllib.parse import urljoin, unquote, urlparse
import xml.etree.ElementTree as ET

from django.core.signing import Signer, BadSignature

from utils.setup_logger import setup_logger

logger = setup_logger(__name__, log_dir="logs/documents_parsing", log_file="webdav_client.log")


class WebDavStorage:
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

    # Остальные методы без изменений
    def list_directory(self, path):
        url = urljoin(self.base_url, path.lstrip('/'))
        headers = {"Depth": "1", "Content-Type": "application/xml"}
        xml = """<?xml version="1.0" encoding="utf-8" ?>
                <d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
                  <d:prop>
                    <d:getlastmodified/>
                    <d:getcontentlength/>
                    <d:resourcetype/>
                    <d:getetag/>
                    <oc:fileid/>
                  </d:prop>
                </d:propfind>
        """
        logger.info(f"PROPFIND запрос к {url}")
        try:
            response = requests.request("PROPFIND", url, data=xml, headers=headers, auth=self.auth)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Ошибка PROPFIND для {url}: {e}")
            logger.debug(f"Тело ответа: {response.text}")
            xml_fallback = """<?xml version="1.0" encoding="utf-8" ?>
                            <d:propfind xmlns:d="DAV:">
                              <d:prop>
                                <d:getlastmodified/>
                                <d:getcontentlength/>
                                <d:resourcetype/>
                                <d:getetag/>
                              </d:prop>
                            </d:propfind>
            """
            logger.info(f"Повторный PROPFIND без oc:fileid к {url}")
            try:
                response = requests.request("PROPFIND", url, data=xml_fallback, headers=headers, auth=self.auth)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                logger.error(f"Повторная ошибка PROPFIND: {e}")
                raise

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

            item = {
                'path': rel_path,
                'file_name': file_name,
                'is_dir': is_dir,
                'last_modified': resp.findtext('d:propstat/d:prop/d:getlastmodified', default=None, namespaces=ns),
                'size': int(resp.findtext('d:propstat/d:prop/d:getcontentlength', default="0",
                                          namespaces=ns)) if not is_dir else 0,
                'etag': resp.findtext('d:propstat/d:prop/d:getetag', default=None, namespaces=ns),
                'url': urljoin(self.base_url, rel_path.lstrip('/')),
                'fileid': resp.findtext('d:propstat/d:prop/oc:fileid', default=None, namespaces=ns)
            }

            if item['fileid']:
                logger.debug(f"Найден file_id: {item['fileid']} для {item['path']}")
            else:
                logger.debug(f"file_id отсутствует для {item['path']}")

            items.append(item)
        return items

    def download_file(self, path):
        url = urljoin(self.base_url, path.lstrip('/'))
        logger.info(f"Скачивание файла: {url}")
        try:
            response = requests.get(url, auth=self.auth)
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            logger.error(f"Ошибка скачивания {url}: {e}")
            raise

    def sync_all(self):
        all_files = []

        def recursive_list(path):
            try:
                items = self.list_directory(path)
            except Exception as e:
                logger.error(f"Ошибка чтения директории {path}: {e}")
                return

            for item in items:
                if item['is_dir']:
                    sub_path = item['path']
                    if sub_path.rstrip('/') == path.rstrip('/'):
                        continue
                    logger.info(f"Обработка поддиректории: {sub_path}")
                    recursive_list(sub_path)
                else:
                    all_files.append(item)

        logger.info(f"Сбор всех файлов из {self.root_path}")
        recursive_list(self.root_path)
        return all_files

    def sync_selected(self, file_paths):
        selected_files = []
        for path in file_paths:
            try:
                url = urljoin(self.base_url, path.lstrip('/'))
                headers = {"Depth": "0", "Content-Type": "application/xml"}
                xml = """<?xml version="1.0" encoding="utf-8" ?>
                        <d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
                          <d:prop>
                            <d:getlastmodified/>
                            <d:getcontentlength/>
                            <d:resourcetype/>
                            <d:getetag/>
                            <oc:fileid/>
                          </d:prop>
                        </d:propfind>
                """
                response = requests.request("PROPFIND", url, data=xml, headers=headers, auth=self.auth)
                response.raise_for_status()

                ns = {'d': 'DAV:', 'oc': 'http://owncloud.org/ns'}
                root = ET.fromstring(response.text)
                resp = root.find('d:response', ns)
                if not resp:
                    logger.warning(f"Файл не найден: {path}")
                    continue

                href_elem = resp.find('d:href', ns)
                href = unquote(href_elem.text)
                rel_path = href.lstrip('/')
                if rel_path.startswith('public.php/webdav/'):
                    rel_path = rel_path[len('public.php/webdav/'):]

                parsed = urlparse(href)
                file_name = parsed.path.rstrip('/').split('/')[-1]

                resourcetype = resp.find('d:propstat/d:prop/d:resourcetype', ns)
                is_dir = resourcetype is not None and resourcetype.find('d:collection', ns) is not None
                if is_dir:
                    logger.warning(f"Путь {path} — директория, пропущена")
                    continue

                item = {
                    'path': rel_path,
                    'file_name': file_name,
                    'is_dir': False,
                    'last_modified': resp.findtext('d:propstat/d:prop/d:getlastmodified', default=None, namespaces=ns),
                    'size': int(resp.findtext('d:propstat/d:prop/d:getcontentlength', default="0", namespaces=ns)),
                    'etag': resp.findtext('d:propstat/d:prop/d:getetag', default=None, namespaces=ns),
                    'url': urljoin(self.base_url, rel_path.lstrip('/')),
                    'fileid': resp.findtext('d:propstat/d:prop/oc:fileid', default=None, namespaces=ns)
                }

                selected_files.append(item)
                logger.info(f"Добавлен файл для синхронизации: {path}")
            except Exception as e:
                logger.error(f"Ошибка обработки файла {path}: {e}")

        return selected_files