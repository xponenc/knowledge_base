import re

MARKDOWN_PATTERN = re.compile(
    r"(\*\*.+?\*\*|__.+?__|`.+?`|#+\s.+|\n[-*]\s|\n\d+\.\s|\[.+?\]\(.+?\))"
)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")


def is_markdown(text: str) -> bool:
    """Простая проверка, содержит ли текст markdown-разметку,
    включая ссылки вида [текст](URL)"""
    return bool(MARKDOWN_PATTERN.search(text))


def format_links_markdown(text: str) -> str:
    """
    Преобразует ссылки в Markdown с выделением текста курсивом,
    не трогая сам URL.
    """
    def repl(match):
        text, url = match.groups()
        return f'*[{text}]({url})*'  # курсив вокруг текста, URL не меняем
    return MARKDOWN_LINK_PATTERN.sub(repl, text)