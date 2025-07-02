from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Глобальный кеш для модели и токенизатора
_model_cache = {}

def get_model_and_tokenizer(model_name='utrobinmv/t5_summary_en_ru_zh_base_2048', device=None):
    global _model_cache
    if 'model' not in _model_cache or 'tokenizer' not in _model_cache:
        _model_cache['model'] = T5ForConditionalGeneration.from_pretrained(model_name)
        _model_cache['model'].eval()
        _model_cache['tokenizer'] = T5Tokenizer.from_pretrained(model_name)
        if device:
            _model_cache['model'].to(device)
    return _model_cache['model'], _model_cache['tokenizer']

def summarize_text(text: str, mode: str = 'summary', device: str = None) -> str:
    """
    Получить суммаризацию текста с помощью модели T5.

    mode: one of 'summary' (обычная), 'summary brief' (краткая), 'summary big' (расширенная)

    device: 'cpu', 'cuda', или None (тогда используется 'cuda' если доступна)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer = get_model_and_tokenizer(device=device)

    prefix = mode.strip().lower()
    if prefix not in ('summary', 'summary brief', 'summary big'):
        raise ValueError(f"Unsupported mode: {mode}. Use one of 'summary', 'summary brief', 'summary big'.")

    src_text = prefix + ': ' + text
    inputs = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_length=200, num_beams=5, early_stopping=True)
    summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Обычно результат - список из одной строки
    return summaries[0] if summaries else ''

# Пример использования:
if __name__ == '__main__':
    text = """Высота башни составляет 324 метра (1063 фута), примерно такая же высота, как у 81-этажного здания..."""
    print("Full summary:")
    print(summarize_text(text, mode='summary'))
    print("\nBrief summary:")
    print(summarize_text(text, mode='summary brief'))
    print("\nBig summary:")
    print(summarize_text(text, mode='summary big'))

import asyncio

import aiohttp


def summarize_with_sber(text: str, beams=5, count=3, length_penalty=0.5) -> str | None:
    payload = {
        "instances": [
            {
                "text": text,
                "num_beams": beams,
                "num_return_sequences": count,
                "length_penalty": length_penalty
            }
        ]
    }

    async def fetch():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.aicloud.sbercloud.ru/public/v2/summarizator/predict",
                json=payload,
                headers={"Content-Type": "application/json", "accept": "application/json"}
            ) as response:
                try:
                    data = await response.json()
                except aiohttp.ContentTypeError:
                    # Если ответ не JSON
                    return {"error": f"Unexpected content type: {response.content_type}"}

                if response.status != 200:
                    return {"error": f"HTTP {response.status}", "response": data}

                return data

    result = asyncio.run(fetch())

    # Проверяем и достаём summary
    try:
        if "prediction_best" in result and "bertscore" in result["prediction_best"]:
            return result["prediction_best"]["bertscore"]
        elif "predictions" in result:
            # fallback — возвращаем первый вариант
            return result["predictions"][0]
        else:
            return None
    except Exception as e:
        return f"Ошибка при извлечении summary: {e}"