from pprint import pprint
from typing import Dict, Any, Iterable, List

from langchain_core.messages import AIMessage


def list_cleaner(lst):
    """Убирает пустые строки, дубликаты, пробелы."""
    return list(dict.fromkeys([item.strip() for item in lst if item.strip()]))


def debug_inputs(_inputs, stage_name=""):
    """
    Отладочная функция для вывода содержимого inputs на определенном этапе цепочки.
    """
    print(f"[Debug {stage_name}] inputs:")
    pprint(_inputs)
    return _inputs


def unpack_original_inputs(_inputs):
    """
    Распаковывает original_inputs, удаляет его из inputs и объединяет с остальными данными.
    """
    original_inputs = _inputs.get("original_inputs", {})
    if "original_inputs" in _inputs:
        _inputs.pop("original_inputs")
    new_inputs = {
        **_inputs,
        **original_inputs
    }
    return new_inputs


# Функция для распаковки результатов sequential_chains
def unpack_sequential(_inputs, debug_mode: bool = False):
    if debug_mode:
        print(f"\n\n[Unpack Sequential] inputs:")
        pprint(_inputs)
    summary_and_router_output = _inputs.get("summary_and_router", {})
    search_index_output = _inputs.get("search_index", [])
    original_inputs = _inputs.get("original_inputs", {})
    new_inputs = {
        **original_inputs,
        **summary_and_router_output,
        **search_index_output
    }
    if debug_mode:
        print(f"[Unpack Sequential] new inputs:")
        pprint(_inputs)

    return new_inputs


def print_dict_structure(d: dict, indent: int = 0):
    """Рекурсивно печатает структуру словаря с типами значений."""
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}/")  # "/" показываем как папку
            print_dict_structure(value, indent + 1)
        else:
            print(f"{prefix}{key}: {type(value).__name__}")


def unpack_inputs(inputs: Dict[str, Any], keys: Iterable[str] = ("original_inputs",)) -> Dict[str, Any]:
    """
    Универсальная функция распаковки вложенных словарей из inputs.

    - Берёт словарь `inputs` (например, из пайплайна).
    - Для каждого ключа в `keys` проверяет, есть ли он в inputs.
    - Если значение по этому ключу — словарь, то он разворачивается и объединяется с inputs.
    - Исходный ключ удаляется из inputs.
    - Возвращает обновлённый словарь inputs.

    ⚠️ Внимание:
        Функция изменяет входной словарь `inputs` (побочный эффект).
        Если нужно сохранить оригинал — передайте копию словаря.

    Args:
        inputs (Dict[str, Any]): Исходный словарь, в котором может быть вложенность.
        keys (Iterable[str], optional): Набор ключей для распаковки.
                                        По умолчанию — только ("original_inputs",).

    Returns:
        Dict[str, Any]: Обновлённый словарь inputs с объединёнными данными.

    Пример:
        >>> data = {
        ...     "original_inputs": {"a": 1, "b": 2},
        ...     "report_and_router": {"c": 3},
        ...     "x": 42
        ... }
        >>> unpack_inputs(data, keys=("original_inputs", "report_and_router"))
        {'x': 42, 'a': 1, 'b': 2, 'c': 3}
    """
    updates: Dict[str, Any] = {}

    # Собираем данные для объединения
    for key in keys:
        value = inputs.get(key)
        if value and isinstance(value, dict):
            updates.update(value)

    # Убираем ключи, чтобы они не дублировались
    for key in keys:
        inputs.pop(key, None)

    # Объединяем в исходный словарь
    inputs.update(updates)

    return inputs


def extract_list(split_data: str) -> List[str]:
    """
    Извлекает список из строки.
    """
    try:
        output = [
            x.strip().strip('"')
            for x in split_data.split(",")
            if x.strip().strip('"')
        ]
    except Exception:
        output = []
    return output


def merge_unique_lists(list1: list[str] | None, list2: list[str] | None) -> list[str]:
    """Объединяет два списка строк в один с уникальными значениями, сохраняя порядок."""
    if not list1:
        list1 = []
    if not list2:
        list2 = []
    seen = set()
    merged = []
    for item in list1 + list2:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged
