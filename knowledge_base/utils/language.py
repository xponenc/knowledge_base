from langdetect import detect

def detect_language(content):
    """
    Определяет язык контента.
    :param content: Текст для анализа.
    :return: Код языка (например, 'en', 'ru').
    """
    try:
        return detect(content)
    except:
        return None