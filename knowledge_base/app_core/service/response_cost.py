PRICE_MODELS = {
    "text_tokens": {
        "gpt-5": {"input": 1.25, "cashed_input": 0.125, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "cashed_input": 0.025, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "cashed_input": 0.005, "output": 0.40},
        "gpt-4.1": {"input": 2.00, "cashed_input": 0.50, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "cashed_input": 0.10, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "cashed_input": 0.025, "output": 0.40},
        "gpt-4o": {"input": 2.50, "cashed_input": 1.25, "output": 10.00},
        "gpt-4o-audio-preview": {"input": 2.50, "cashed_input": "", "output": 10.00},
        "gpt-4o-realtime-preview": {"input": 5.00, "cashed_input": 2.50, "output": 20.00},
        "gpt-4o-mini": {"input": 0.15, "cashed_input": 0.075, "output": 0.60},
        "gpt-4o-mini-audio-preview": {"input": 0.15, "cashed_input": "", "output": 0.60},
        "gpt-4o-mini-realtime-preview": {"input": 0.60, "cashed_input": 0.30, "output": 2.40},
        "o1": {"input": 15.00, "cashed_input": 7.50, "output": 60.00},
        "o1-pro": {"input": 150.00, "cashed_input": "", "output": 600.00},
        "o3-pro": {"input": 20.00, "cashed_input": "", "output": 80.00},
        "o3": {"input": 2.00, "cashed_input": 0.50, "output": 8.00},
        "o3-deep-research": {"input": 10.00, "cashed_input": 2.50, "output": 40.00},
        "o4-mini": {"input": 1.10, "cashed_input": 0.275, "output": 4.40},
        "o4-mini-deep-research": {"input": 2.00, "cashed_input": 0.50, "output": 8.00},
        "o3-mini": {"input": 1.10, "cashed_input": 0.55, "output": 4.40},
        "o1-mini": {"input": 1.10, "cashed_input": 0.55, "output": 4.40},
        "codex-mini-latest": {"input": 1.50, "cashed_input": 0.375, "output": 6.00},
        "gpt-4o-mini-search-preview": {"input": 0.15, "cashed_input": "", "output": 0.60},
        "gpt-4o-search-preview": {"input": 2.5, "cashed_input": "", "output": 10.00},
        "computer-use-preview": {"input": 3.00, "cashed_input": "", "output": 12.00},
        "gpt-image-1": {"input": 5.00, "cashed_input": 1.25, "output": ""},
    }
}


def get_price(prompt_token_counter: int,
              answer_token_counter:int,
              model_name: str,
              mode: str = "text_tokens",
              cashed: bool = False):
    """Возвращает стоимость обработки вопроса LLM"""
    inf = float('inf')

    model_price = PRICE_MODELS.get(mode, {}).get(model_name)

    if not model_price:

        return inf, inf, inf

    prompt_price = model_price.get("input")
    if cashed:
        answer_price = model_price.get("cashed_input")
    else:
        answer_price = model_price.get("output")

    try:
        prompt_cost = prompt_price * prompt_token_counter / 1_000_000
    except ValueError:
        prompt_cost = inf

    try:
        answer_cost = answer_price * answer_token_counter / 1_000_000
    except ValueError:
        answer_cost = inf

    total_cost = answer_cost + prompt_cost

    return total_cost, prompt_cost, answer_cost


def format_cost(value):
    if value == float('inf'):
        return "∞"
    if value == 0:
        return "0"
    if value < 0.0001:  # меньше 0.0001 — научная запись
        return f"{value:.2e}"
    return f"{value:.6f}".rstrip('0').rstrip('.')