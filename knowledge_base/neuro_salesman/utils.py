def list_cleaner(lst):
    """Убирает пустые строки, дубликаты, пробелы."""
    return list(dict.fromkeys([item.strip() for item in lst if item.strip()]))