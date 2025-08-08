from roles_config import ROLES
from extractor import run_extractor
from llm_utils import call_llm
from utils import list_cleaner

def get_seller_answer(user_message, histories, neuro_data, verbose=False):
    """
    Генерация ответа менеджера с участием нескольких экспертов.
    """
    # 1. Экстракторы
    extractors = {
        "needs": "needs_extractor",
        "benefits": "benefits_extractor",
        "objections": "objection_detector",
        "resolved_objections": "resolved_objection_detector",
        "tariffs": "tariff_extractor"
    }

    for key, extractor_key in extractors.items():
        worker = ROLES[extractor_key]
        extracted = run_extractor(
            role_config=worker,
            question=user_message if key in ["needs", "objections"] else "",
            history=histories.get("manager", []),
            verbose=verbose
        )
        if extracted:
            neuro_data[key] = list_cleaner(neuro_data.get(key, []) + [extracted])

    # 2. Сводка диалога
    summary = call_llm(
        ROLES["summary"]["system_prompt"],
        f"{ROLES['summary']['instructions']}\n\nИстория диалога:\n" + "\n".join(histories.get("dialogue", [])),
        temp=ROLES["summary"]["temperature"],
        verbose=verbose
    )

    # 3. Роутер — теперь выдаёт список экспертов
    experts_str = call_llm(
        ROLES["router"]["system_prompt"],
        (
            "Проанализируй диалог и выбери из списка экспертов всех, кто может помочь.\n"
            "Доступные эксперты: Маркетолог, Продуктовый эксперт, Технический специалист, Юрист, Финансовый эксперт.\n"
            "Ответь через запятую."
            f"\n\nСводка диалога:\n{summary}"
        ),
        temp=ROLES["router"]["temperature"],
        verbose=verbose
    )

    experts = [e.strip() for e in experts_str.split(",") if e.strip() in ROLES]

    # 4. Запрашиваем советы у каждого выбранного эксперта
    expert_advices = []
    for expert_name in experts:
        advice = call_llm(
            ROLES[expert_name]["system_prompt"],
            f"{ROLES[expert_name]['instructions']}\n\nЗапрос клиента:\n{user_message}\n\nСводка диалога:\n{summary}",
            temp=ROLES[expert_name]["temperature"],
            verbose=verbose
        )
        expert_advices.append(f"{expert_name}: {advice}")

    # 5. Старший менеджер — собирает ответ с учётом всех советов
    draft_answer = call_llm(
        ROLES["senior_manager"]["system_prompt"],
        f"{ROLES['senior_manager']['instructions']}\n\nСводка:\n{summary}\n\nСоветы экспертов:\n" + "\n".join(expert_advices),
        temp=ROLES["senior_manager"]["temperature"],
        verbose=verbose
    )

    # 6. Стилист — финальная правка
    final_answer = call_llm(
        ROLES["stylist"]["system_prompt"],
        f"{ROLES['stylist']['instructions']}\n\n{draft_answer}",
        temp=ROLES["stylist"]["temperature"],
        verbose=verbose
    )

    return final_answer
