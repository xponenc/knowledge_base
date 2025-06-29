import re

import emoji


def remove_emoji(text):
    """
    Удаляет эмоджи из входного текста, используя комбинацию библиотеки emoji и регулярных выражений.

    Функция сначала применяет библиотеку emoji для удаления известных эмоджи, а затем использует
    регулярное выражение для удаления оставшихся символов Unicode, относящихся к эмоджи. Этот
    двухэтапный подход обеспечивает полное удаление эмоджи, включая стандартные и нестандартные символы.

    Аргументы:
        text (str): Входной текст, содержащий эмоджи для удаления.

    Возвращает:
        str: Текст с удалёнными эмоджи.

    """
    # Шаг 1: Удаление эмоджи с помощью библиотеки emoji
    text = emoji.replace_emoji(text, replace='')

    # Шаг 2: Удаление оставшихся эмоджи с помощью регулярного выражения
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Эмоции
        "\U0001F300-\U0001F5FF"  # Символы и пиктограммы
        "\U0001F680-\U0001F6FF"  # Транспорт и карты
        "\U0001F700-\U0001F77F"  # Алхимические символы
        "\U0001F780-\U0001F7FF"  # Расширенные геометрические фигуры
        "\U0001F800-\U0001F8FF"  # Дополнительные символы
        "\U0001F900-\U0001F9FF"  # Дополнительные символы и пиктограммы
        "\U0001FA00-\U0001FA6F"  # Шахматные символы
        "\U0001FA70-\U0001FAFF"  # Расширенные символы и пиктограммы
        "\U00002700-\U000027BF"  # Дингбаты
        "\U00002600-\U000026FF"  # Прочие символы
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


# Функция для очистки текста
def normalize_text(text):
    import unicodedata, re

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Заменить неразрывные и нулевые пробелы
    text = text.replace('\u00A0', ' ').replace('\u200B', '')

    # Удалить управляющие и невидимые символы
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'[\u2028\u2029\uFEFF]', '', text)

    # Убрать множественные пробелы
    text = re.sub(r'[ \t]+', ' ', text)

    # Удалить множественные пустые строки
    text = re.sub(r'\n\s*\n', '\n', text)

    # Убрать пустые строки
    text = '\n'.join(line for line in text.split('\n') if line.strip())
    return text


def clean_markdown(text: str) -> str:
    # ШУдаление остатков Markdown-разметки
    # Удаление заголовков (#, ##, ...), списков (*, -, 1.), жирного/курсива
    text = re.sub(r'^(#+|\*|-|\d+\.)\s+.*\n?', '', text, flags=re.MULTILINE)
    # Удаление разделительных линий (---, ___, ***)
    text = re.sub(r'[-*_]{2,}\n?', '', text)
    # Удаление inline-форматирования (**жирный**, *курсив*, `код`)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Удаление ссылок [текст](url) и остатков URL
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'https?://[^\s]+', '', text)
    # Удаление кодовых блоков (```...```)
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    return text
