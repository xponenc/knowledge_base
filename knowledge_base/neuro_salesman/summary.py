from typing import Dict

from langchain_core.messages import AIMessage


def create_summary_exact(inputs: Dict) -> Dict:
    """Создаёт точное саммари диалога на основе результатов экстракторов."""
    needs = inputs.get("needs", AIMessage(content="")).content.split(",") if inputs.get("needs").content else []
    needs = [n.strip() for n in needs if n.strip()]
    benefits = inputs.get("benefits", AIMessage(content="")).content.split(",") if inputs.get("benefits").content else []
    benefits = [b.strip() for b in benefits if b.strip()]
    objections = inputs.get("objections", AIMessage(content="")).content.split(",") if inputs.get("objections").content else []
    objections = [o.strip() for o in objections if o.strip()]
    resolved_objections = inputs.get("resolved_objections", AIMessage(content="")).content.split(",") if inputs.get("resolved_objections").content else []
    resolved_objections = [r.strip() for r in resolved_objections if r.strip()]
    tariffs = inputs.get("tariffs", AIMessage(content="")).content.split(",") if inputs.get("tariffs").content else []
    tariffs = [t.strip() for t in tariffs if t.strip()]

    summary_exact = (
        f"# 1. Выявлены Потребности: {', '.join(needs) if needs else 'потребностей не обнаружено'}\n"
        f"# 2. Рассказанные Преимущества: {', '.join(benefits) if benefits else 'преимущества не были рассказаны'}\n"
        f"# 3. Возражения клиента: {', '.join(objections) if objections else 'возражений не обнаружено'}\n"
        f"# 4. Возражения клиента отработаны: {', '.join(resolved_objections) if resolved_objections else 'отработки не обнаружено'}\n"
        f"# 5. Конкретика - оговоренная конкретика - курсы, цены: {', '.join(tariffs) if tariffs else 'не обнаружено'}\n"
    )

    return {**inputs, "summary_exact": summary_exact}