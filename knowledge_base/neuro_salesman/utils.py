from pprint import pprint


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
def unpack_sequential(_inputs, debug_mode:bool = False):
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

