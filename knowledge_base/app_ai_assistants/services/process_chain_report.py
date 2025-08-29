from typing import Any, Dict
import re

from langchain_core.messages import AIMessage

from app_core.service.response_cost import get_price


def process_chain_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обрабатывает результаты ответов цепочек:
    - Формирует структуру данных по каждой цепочке
    - Считает итоговую стоимость
    """

    summary: Dict[str, Any] = {
        "chains": {},
        "total": {
            "cost": 0.0,
            "prompt_tokens": 0,
            "answer_tokens": 0,
        }
    }

    for chain_name, chain_result in results.items():
        if isinstance(chain_result, AIMessage):
            content = chain_result.content

            # Токены и модель
            metadata = chain_result.response_metadata
            prompt_tokens = metadata.get("token_usage", {}).get("prompt_tokens", 0)
            answer_tokens = metadata.get("token_usage", {}).get("completion_tokens", 0)
            model = metadata.get("model_name", "")

            # Чистим модель от даты
            model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)

            # Стоимость
            total_cost, prompt_cost, answer_cost = get_price(
                prompt_token_counter=prompt_tokens,
                answer_token_counter=answer_tokens,
                model_name=model,
            )

            # Записываем в словарь по цепочке
            summary["chains"][chain_name] = {
                "content": content,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
                "cost": {
                    "total": total_cost,
                    "prompt": prompt_cost,
                    "answer": answer_cost,
                }
            }

            # Итоговые суммы
            summary["total"]["cost"] += total_cost
            summary["total"]["prompt_tokens"] += prompt_tokens
            summary["total"]["answer_tokens"] += answer_tokens

        else:
            # Если результат не AIMessage — просто сохраняем как есть
            summary["chains"][chain_name] = {
                "content": str(chain_result),
                "model": None,
                "prompt_tokens": 0,
                "answer_tokens": 0,
                "cost": {
                    "total": 0.0,
                    "prompt": 0.0,
                    "answer": 0.0,
                }
            }

    return summary
