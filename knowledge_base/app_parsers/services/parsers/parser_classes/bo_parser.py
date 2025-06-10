import re
from typing import Dict, Any, List, Tuple, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag, NavigableString, Comment

from app_parsers.services.parsers.base import BaseWebParser


class BOWebParser(BaseWebParser):
    BREADCRUMBS_CLASS = (".breadcrumbs a.taxonomy.category")

    EXCLUDE_TAGS = [
        "footer, header, nav, menu, sidebar, popup, modal, banner, ad, subscribe, widget, cookie, social, share, logo, script, style, form, input, iframe, svg, noscript, button, select, option, canvas, link, meta, jdiv"
    ]
    EXCLUDE_id = "preload"
    EXCLUDE_CLASSES = (
        "header__top, coast_block, express_test_marquiz, order_tel, cf7_form, sw-app, modal, calc, category__info-sale, breadcrumbs, my_yandex_centered_text")
    STYLE_TAGS = {'strong, b, i, em, u, span'}
    useful_image_classes = ["attachment-full, size-full"]

    config_schema = {
        "title_tag": {
            "type": str,
            "label": "Тег заголовка страницы",
            "help_text": "Введите HTML-тег, в котором содержится заголовок (например: h1, div, span)."
                         " Если оставить пустым — будет выбран h1 или h2 автоматически",
        },
        "title_class": {
            "type": str,
            "label": "CSS-класс заголовка",
            "help_text": "Введите имя CSS-класса, если заголовок находится в конкретном классе. Можно оставить пустым.",
        },
        "breadcrumbs_selector": {
            "type": str,
            "label": "CSS-селектор хлебных крошек",
            "help_text": (
                "Укажите CSS-селектор для извлечения ссылок категорий из навигации (хлебных крошек).\n"
                "Формат: .класс_родителя тег.класс_элемента\n"
                "Например: `.breadcrumbs a.taxonomy.category` — выбирает все ссылки <a> с классами "
                "`taxonomy` и `category`, которые находятся внутри элемента с классом `breadcrumbs`.\n"
                "Если вы не уверены — оставьте значение по незаполненным, по умолчанию будет выполнен поиск в .breadcrumbs a"
            ),
        },
        "exclude_tags": {
            "type": list[str],
            "label": "Удаляемые HTML теги",
            "help_text": "Вводите названия тегов по одному через ',' или ';' или"
                         " перевод строки (например: script, style)"
        },
        "exclude_ids": {
            "type": list[str],
            "label": "Удаляемые элементы с id",
            "help_text": "Вводите названия тегов по одному через ',' или ';' или"
                         " перевод строки (например: preload, map)"
        },
        "exclude_classes": {
            "type": list[str],
            "label": "Удаляемые CSS классы",
            "help_text": "Вводите названия CSS классов по одному через ',' или ';' или "
                         "перевод строки (например: header__top; coast_block; express_test_marquiz)"
        },

        "ignore_style_tags": {
            "type": list[str],
            "label": "Игнорируемые при обработке стилевые теги, игнорируются при обработке"
                     ", из них просто забирается текст",
            "help_text": "Вводите названия CSS классов по одному через ',' или ';' или"
                         " перевод строки (например: strong, b, i)"
        },
        "useful_image_classes": {
            "type": list[str],
            "label": "Классы изображений, которые будут сохранены при парсинге",
            "help_text": "Вводите названия CSS классов по одному через ',' или ';' или"
                         " перевод строки (например: attachment-full, size-full)"
        },
        "remove_inter_page_links": {
            "type": bool,
            "label": "Удалять в контенте ссылки на другие страницы сайта",
            "help_text": "Введите bool Значение (True/False, 1/0)"
        }
    }

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)

    def parse_html(self,
                   url: str,
                   html: str) -> Dict:
        """
        Основной метод, который должен возвращать:
        {
            "title": str, # заголовок основного контента страницы
            "tags": list[str], # список тэгов (категорий) страницы
            "content": str, # основной контент страницы
            "metadata": dict # словарь с метаданными страницы
        }
        """
        result = self._extract_article_data(url=url, html=html)
        return result

    @staticmethod
    def _extract_main_content(soup: BeautifulSoup):
        """
        Извлекает основной контент страницы на основе семантических тегов.

        :param soup: Объект BeautifulSoup с HTML-контентом.
        :return: Тег, содержащий основной контент.
        """
        # candidates = soup.find_all(['article', 'main', 'section', 'div'], recursive=True)
        # main = max(candidates, key=lambda tag: len(tag.find_all(['p', 'h1', 'h2', 'h3', 'ul'])), default=soup.body)

        main = soup.find('main')
        if not main:
            main = soup.body
        return main

    def _clean_soup(self, soup: BeautifulSoup, url: str) -> BeautifulSoup:
        """
        Очищает HTML-содержимое от ненужных тегов, классов и элементов.

        :param soup: Объект BeautifulSoup с HTML-контентом.
        :param url: URL страницы, используется для преобразования относительных ссылок.
        :return: Очищенный объект BeautifulSoup.
        """

        EXCLUDE_TAGS = self.config.get("exclude_tags")
        EXCLUDE_CLASSES = self.config.get("exclude_classes")
        EXCLUDE_IDS = self.config.get("exclude_ids")

        # Удаление тегов из EXCLUDE_TAGS (например, скрипты, формы)
        if EXCLUDE_TAGS:
            for tag in soup(EXCLUDE_TAGS):
                tag.decompose()
        if EXCLUDE_CLASSES:
            print(f"{EXCLUDE_CLASSES=}")
            # Удаление элементов с классами из EXCLUDE_CLASSES
            # for element in soup.find_all(class_=EXCLUDE_CLASSES):
            #     print(element)
            #     element.decompose()
            for element in soup.find_all(lambda _tag: any(
                    cls in EXCLUDE_CLASSES
                    for cls in _tag.get("class", [])
            )):
                element.decompose()
        if EXCLUDE_IDS:
            # Удаление элементов с ID, содержащими ключевые слова из EXCLUDE_KEYWORDS
            # for el in soup.find_all(attrs={"id": True}):
            #     if any(k in el['id'].lower() for k in EXCLUDE_IDS):
            #         el.decompose()
            for el in soup.find_all(lambda _tag: _tag.has_attr("id") and _tag["id"].lower() in EXCLUDE_IDS):
                el.decompose()

        # Удаление HTML-комментариев
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        # Удаление скрытых элементов
        for tag in soup.select('div[style*="display:none"]'):
            tag.decompose()

        return soup

    def _process_images(
            self,
            soup: BeautifulSoup,
            url: str,
            data_src_url_name: str = "data-src",
            clear_img: bool = False,
    ) -> BeautifulSoup:
        """
        Нормализует HTML-содержимое от ненужных тегов, классов и элементов.

        :param data_src_url_name: название параметра у <img> где хранится ссылка на полноразмерного изображение
        :param clear_img: удалять изображение без data_src_url_name
        :param soup: Объект BeautifulSoup с HTML-контентом.
        :param url: URL страницы, используется для преобразования относительных ссылок.
        :return: Очищенный объект BeautifulSoup.
        """
        useful_image_classes = self.config.get("useful_image_classes")
        for img in soup.find_all("img"):
            data_src = img.get(data_src_url_name)
            src = img.get("src")
            if clear_img and not (data_src or any(img_cls in useful_image_classes for img_cls in img.get("class", []))):
                img.decompose()
                continue
            if src and src.startswith("/"):
                src = f"{url}{src}"
                img["src"] = src
            if data_src and data_src.startswith("/"):
                data_src = f"{url}{data_src}"
                img["data-src"] = data_src
        return soup

    def _find_title_tag(self, soup: BeautifulSoup) -> str:
        """Возвращает текст заголовка страницы"""
        title_tag_name = self.config.get("title_tag_name")
        title_class_name = self.config.get("title_class_name")
        if title_tag_name:
            selector = title_tag_name
            if title_class_name:
                selector += f".{title_class_name}"
            tag = soup.select_one(selector)
        else:
            tag = soup.select_one("h1") or soup.select_one("h2")

        return tag.get_text(strip=True) if tag else None

    def _extract_breadcrumb_categories(self, soup: BeautifulSoup) -> list[str]:
        """Извлекает список тегов страницы из хлебных крошек"""
        selector = self.config.get("breadcrumbs_selector", ".breadcrumbs a")
        elements = soup.select(selector)
        return [el.get_text(strip=True) for el in elements if el.get_text(strip=True)]

    @staticmethod
    def _extract_and_remove_internal_links(soup: BeautifulSoup, url: str) -> Tuple[BeautifulSoup, Set[Tuple[str, str]]]:
        """
        Извлекает все внутренние HTML-ссылки со страницы, удаляет их из soup и возвращает set кортежей (текст ссылки, URL).

        Args:
            soup (BeautifulSoup): Объект BeautifulSoup с разобранным HTML.
            url (str): Базовый URL страницы для определения внутренних ссылок.

        Returns:
            Tuple[BeautifulSoup, Set[Tuple[str, str]]]: Кортеж, содержащий обновлённый soup и множество кортежей (текст ссылки, URL).
        """
        internal_links = set()
        base_domain = urlparse(url).netloc

        # Расширения, которые считаем НЕ HTML-страницами
        non_html_extensions = (
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.rar', '.7z', '.tar', '.gz', '.mp4', '.mp3', '.exe',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.ico',
            '.csv', '.json', '.xml'
        )

        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            absolute_url = urljoin(url, href)
            parsed_link = urlparse(absolute_url)
            link_text = anchor.get_text(strip=True) or absolute_url

            # Убираем якоря и параметры
            path = parsed_link.path.lower()
            if parsed_link.netloc == base_domain and not path.endswith(non_html_extensions):
                internal_links.add((link_text, absolute_url))
                anchor.decompose()

        return soup, internal_links

    @staticmethod
    def _process_http_links(
            soup: BeautifulSoup,
            url: str,
            clear_link_anchor: bool = True,
    ) -> BeautifulSoup:
        """
        Нормализует ссылки в абсолютные

        :param clear_link_anchor: удалять ссылки на якоря внутри страницы
        :param soup: Объект BeautifulSoup с HTML-контентом.
        :param url: URL страницы, используется для преобразования относительных ссылок.
        :return: Очищенный объект BeautifulSoup.
        """
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("#") and clear_link_anchor:
                a.decompose()  # Удаление ссылок, ведущих на внутренние страницы
                continue
            full_url = urljoin(url, href)
            a["href"] = full_url

        return soup

    @staticmethod
    def _extract_images(soup: BeautifulSoup) -> List[Tuple[str, str]]:
        """
        Извлекает все изображения с элемента BeautifulSoup.

        Args:
            soup (BeautifulSoup): Объект BeautifulSoup, содержащий очищенный контент страницы.

        Returns:
            List[Tuple[str, str]]: Список кортежей (alt-текст, URL изображения).
        """
        page_images = []
        for img in soup.find_all("img"):
            src = img.get("data-src") or img.get("src")  # Получаем src или data-src изображения
            alt = img.get("alt", "")  # Получаем alt-текст изображения
            if src:
                page_images.append((alt, src))  # Добавляем изображение как кортеж (alt, src)

        return page_images

    @staticmethod
    def _extract_document_links(soup: BeautifulSoup, base_url: str) -> List[Tuple[str, str]]:
        """
        Извлекает все ссылки на документы (не изображения) с страницы.

        Args:
            soup (BeautifulSoup): Объект BeautifulSoup с разобранным HTML.
            base_url (str): Базовый URL страницы для преобразования относительных ссылок.

        Returns:
            List[Tuple[str, str]]: Список кортежей (текст ссылки, абсолютный URL документа).
        """
        document_links = []
        # Список расширений документов (можно расширить по необходимости)
        document_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.rtf'}

        for anchor in soup.find_all('a', href=True):
            href = anchor.get('href')
            if href:
                # Преобразуем относительную ссылку в абсолютную
                absolute_url = urljoin(base_url, href)
                # Извлекаем расширение файла из URL
                parsed_url = absolute_url.lower().rsplit('?', 1)[0]  # Удаляем параметры запроса
                file_extension = '.' + parsed_url.rsplit('.', 1)[-1] if '.' in parsed_url else ''

                # Проверяем, является ли ссылка на документ (не изображение)
                if file_extension in document_extensions and not any(
                        img_ext in parsed_url for img_ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}):
                    link_text = anchor.get_text(strip=True) or absolute_url  # Текст ссылки или URL, если текста нет
                    document_links.append((link_text, absolute_url))

        return document_links

    @staticmethod
    def _extract_external_links(soup: BeautifulSoup, base_url: str) -> List[Tuple[str, str]]:
        """
        Извлекает все внешние ссылки с страницы.

        Args:
            soup (BeautifulSoup): Объект BeautifulSoup с разобранным HTML.
            base_url (str): Базовый URL страницы для определения внешних ссылок.

        Returns:
            List[Tuple[str, str]]: Список кортежей (текст ссылки, абсолютный URL внешнего ресурса).
        """
        external_links = []
        base_domain = urlparse(base_url).netloc  # Извлекаем домен базового URL

        for anchor in soup.find_all('a', href=True):
            href = anchor.get('href')
            if href:
                # Преобразуем относительные ссылки в абсолютные
                absolute_url = urljoin(base_url, href)
                parsed_link = urlparse(absolute_url)

                # Проверяем, является ли ссылка внешней (другой домен)
                if parsed_link.netloc and parsed_link.netloc != base_domain:
                    link_text = anchor.get_text(strip=True) or absolute_url  # Текст ссылки или URL, если текста нет
                    external_links.append((link_text, absolute_url))
                    # Опционально: удаление тега из soup
                    # anchor.decompose()

        return external_links

    @staticmethod
    def _parse_line(line):
        """
        Разбирает строку из структурированного вывода на уровень, тег и содержимое.

        :param line: Строка из структурированного вывода.
        :return: Кортеж (уровень, тег, содержимое) или None, если строка не соответствует формату.
        """
        # if re.match(r"^-+|.*|", line):
        if re.match(r"^-+\|.*\|", line):
            level = len(re.match(r"^-+", line).group(0))
            return level, "table_row", line.strip('-').strip()

        # match = re.match(r"(-+)(\[a-zA-Z0-9]+):?\s*(.\*)", line)
        match = re.match(r"(-+)([a-zA-Z0-9]+):?\s*(.*)", line)
        if not match:
            return None
        level = len(match.group(1))
        tag = match.group(2).lower()
        content = match.group(3).strip()
        return level, tag, content

    @staticmethod
    def _convert_table_to_markdown(table_tag: Tag) -> List[str]:
        """
        Преобразует HTML-таблицу в формат Markdown.

        :param table_tag: Тег таблицы.
        :return: Список строк с Markdown-представлением таблицы.
        """
        rows = []

        # Извлечение заголовков таблицы
        headers = [th.get_text(strip=True) for th in table_tag.find_all("tr")[0].find_all("th")]
        if headers:
            header_line = "| " + " | ".join(headers) + " |"
            separator_line = "| " + " | ".join(['---'] * len(headers)) + " |"
            rows.append(header_line)
            rows.append(separator_line)

        # Извлечение строк таблицы
        for tr in table_tag.find_all("tr"):
            tds = tr.find_all("td")
            if tds:
                row_line = "| " + " | ".join(td.get_text(strip=True) for td in tds) + " |"
                rows.append(row_line)

        return rows

    def _analyze_element(self, element, level=0, parent_text=None):
        """
        Рекурсивно анализирует элементы HTML и преобразует их в структурированный формат.

        :param element: Текущий элемент HTML.
        :param level: Уровень вложенности для отступов.
        :param parent_text: Список для добавления текста родительского элемента.
        :return: Список строк с описанием элемента.
        """
        STYLE_TAGS = self.config.get("style_tags", [])
        output_lines = []

        if isinstance(element, NavigableString):
            return []  # Пропускаем текстовые узлы без тегов

        if element.name and element.name.startswith(':'):
            return []  # Пропускаем псевдоэлементы

        if element.name == "table":
            markdown_rows = self._convert_table_to_markdown(element)
            indent = '----' * level
            output_lines.append('')  # Разделитель перед таблицей для читаемости
            output_lines.extend([f"{indent}{row}" for row in markdown_rows])
            output_lines.append('')  # Разделитель после таблицы
            return output_lines

        is_style_tag = element.name in STYLE_TAGS

        has_non_style_tags = any(
            isinstance(child, Tag) and not child.name.startswith(':') and child.name not in STYLE_TAGS
            for child in element.children
        )
        # Проверяем, есть ли среди детей теги, которые не являются стилистическими

        direct_text = ' '.join(
            str(child).strip() for child in element.children if isinstance(child, NavigableString)
        ).strip()
        # Собираем текст из текстовых узлов, удаляя лишние пробелы

        child_texts = []

        for child in element.children:
            if isinstance(child, Tag):
                child_result = self._analyze_element(child, level + (0 if is_style_tag else 1), child_texts)
                output_lines.extend(child_result)

        combined_text = ' '.join(filter(None, [direct_text] + child_texts)).strip()
        # Объединяем текст из детей и текущего элемента, фильтруя пустые строки

        if not is_style_tag:
            indent = '----' * level

            if element.name == "a":
                href = element.get("href", "").strip()
                link_text = combined_text if combined_text else href
                md_link = f"[{link_text}]({href})"
                if parent_text is not None:
                    parent_text.append(md_link)
                else:
                    output_lines.insert(0, f"{indent}{md_link}")

            elif element.name == "img":
                src = element.get("data-src") or element.get("src", "")
                alt = element.get("alt", "").strip()
                md_image = f"![{alt}]({src})"
                if parent_text is not None:
                    parent_text.append(md_image)
                else:
                    output_lines.insert(0, f"{indent}{md_image}")

            elif not has_non_style_tags:
                if combined_text:
                    output_lines.insert(0, f"{indent}{element.name}: {combined_text}")
                else:
                    output_lines.insert(0, f"{indent}{element.name}")

            elif combined_text:
                output_lines.insert(0, f"{indent}{element.name}: {combined_text}")
            else:
                output_lines.insert(0, f"{indent}{element.name}")

        elif combined_text and parent_text is not None:
            parent_text.append(combined_text)

        return output_lines

    def _to_markdown(self, lines):
        """
        Преобразует структурированный список строк в формат Markdown.

        :param lines: Список строк из структурированного вывода.
        :return: Строка с Markdown-контентом.
        """
        md_lines = []
        ul_stack = []  # Стек для отслеживания уровней списков ul
        li_counters = 0  # Счетчики для нумерации элементов li
        in_table = False  # Флаг для отслеживания нахождения внутри таблицы

        for line in lines:
            parsed = self._parse_line(line)
            if not parsed:
                continue
            level, tag, content = parsed
            # Отслеживание вложенности списков ul
            if tag == "ul" or tag == "ol":
                ul_stack.append(level)
                continue
            if ul_stack:
                ul_stack = list(filter(lambda item: item < level, ul_stack))
                if not ul_stack:
                    li_counters = 0
                # print(f"[!] {ul_stack=} {level=}, {tag=}, {content=} {li_counters=}")

            if tag == "table_row":
                current_ul_level = len(ul_stack)
                indent = " " * 4 * current_ul_level
                if not in_table:
                    md_lines.append("")  # Пустая строка перед первой строкой таблицы
                    in_table = True
                md_lines.append(f"{indent}{content}")
            else:
                # Сброс флага таблицы при встрече не-строки таблицы
                in_table = False

            if tag == "li":
                # Определение текущего уровня списка ul
                current_ul_level = len(ul_stack)
                # print(ul_stack, level, current_ul_level)
                if current_ul_level == 1:
                    # Верхний уровень — нумерованный список
                    li_counters += 1
                    prefix = f"{li_counters}."
                    md_lines.append(f"{prefix} {content}")
                elif level > 1:
                    # Вложенный ul — ненумерованный список
                    indent = " " * 4 * (current_ul_level - 1)
                    md_lines.append(f"{indent}- {content}")
                else:
                    md_lines.append(f"- {content}")
                    continue
            elif tag == "a":
                match = re.match(r"\[(.+?)\]\((http.*?)\)", content)
                if match:
                    md_lines.append(f"[{match.group(1)}]({match.group(2)})")
                else:
                    md_lines.append(content)
            elif tag == "img":
                match = re.match(r"!\[(.*?)\]\((.*?)\)", content)
                if match:
                    md_lines.append(f"![{match.group(1)}]({match.group(2)})")
                else:
                    md_lines.append(f"![Image]({content})")
            # elif tag == "table_row":
            #     current_ul_level = len(ul_stack)
            #     indent = " " * 4 * current_ul_level
            #     md_lines.append("")
            #     md_lines.append(f"{indent}{content}")

            elif tag in {"h1", "h2", "h3", "h4"}:
                hashes = "#" * int(tag[1])
                md_lines.append(f"{hashes} {content}")
            else:
                if content and tag != "table_row":
                    md_lines.append(content)

        return "\n".join(md_lines)

    def _extract_article_data(self, html: str, url: str) -> Dict[str, Any]:
        """
        Извлекает данные статьи: категории, контент в markdown и изображения.

        :param html: HTML-содержимое страницы, которое будет парситься.
        :param url: URL страницы, используется для контекста и возможных преобразований ссылок.
        :return: Словарь с категориями, контентом в формате Markdown и изображениями.
        """
        internal_links = set()
        parsed = urlparse(url)

        BASE_URL = f"{parsed.scheme}://{parsed.netloc}"

        soup = BeautifulSoup(html, "html.parser")

        page_title = self._find_title_tag(soup)

        # Извлечение категорий из breadcrumbs
        page_tags = self._extract_breadcrumb_categories(soup)

        # Поиск основного контента
        content_element = self._extract_main_content(
            soup)  # extract_main_content выбирает тег с наибольшим количеством контента
        # print(f"{content_element=}")
        cleaned_content_element = self._clean_soup(soup=content_element,
                                                   url=url)  # clean_soup очищает HTML от ненужных элементов

        # Очистка от внутрисайтовых ссылок
        remove_inter_page_links = self.config.get("remove_inter_page_links", False)
        if remove_inter_page_links:
            cleaned_content_element, internal_links = self._extract_and_remove_internal_links(
                soup=cleaned_content_element, url=url)

        # clean_soup очищает HTML от ненужных изображений
        cleaned_content_element = self._process_images(soup=cleaned_content_element, url=BASE_URL,
                                                       clear_img=True)

        # Очистка от внутристраничных ссылок
        cleaned_content_element = self._process_http_links(soup=cleaned_content_element, url=url,
                                                           clear_link_anchor=False)
        # print(f"{cleaned_content_element=}")

        # Преобразование контента в структурированный формат
        html_structure = self._analyze_element(cleaned_content_element,
                                               0)  # analyze_element рекурсивно разбирает HTML-структуру
        # print(f"{html_structure=}")
        # Преобразование структуры в Markdown
        markdown_content = self._to_markdown(
            html_structure)  # to_markdown преобразует структурированный формат в Markdown
        # pprint(f"{markdown_content=}")

        # Извлечение изображений
        page_images = self._extract_images(soup=cleaned_content_element)
        # Извлечение документов
        page_documents = self._extract_document_links(soup=cleaned_content_element, base_url=BASE_URL)
        # Извлечение ссылок на внешние источники
        page_external_links = self._extract_external_links(soup=cleaned_content_element, base_url=BASE_URL)

        # logger.info(f"Успешно извлечено {len(markdown_content)} символов контента для URL: {url}")
        return {
            "content": markdown_content.strip(),
            "metadata": {
                "title": page_title,
                "tags": page_tags,
                "files": {
                    "images": page_images,
                    "documents": page_documents,
                },
                "external_links": page_external_links,
                "internal_links": tuple(internal_links),
            }
        }
