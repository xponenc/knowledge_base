from typing import Dict

from langchain_core.messages import AIMessage

from neuro_salesman.chains.chain_logger import ChainLogger

#
# def create_summary_exact(inputs: Dict, debug_mode: bool = False) -> Dict:
#     """Создаёт точное саммари диалога на основе результатов экстракторов."""
#     print(f"Summary {inputs=}")
#     needs = inputs.get("needs", AIMessage(content="")).content.split(",") if inputs.get("needs").content else []
#     needs = [n.strip() for n in needs if n.strip()]
#
#     benefits = (inputs.get("benefits", AIMessage(content="")).content.split(",")
#                 if inputs.get("benefits").content else [])
#     benefits = [b.strip() for b in benefits if b.strip()]
#
#     objections = (inputs.get("objections", AIMessage(content="")).content.split(",")
#                   if inputs.get("objections").content else [])
#     objections = [o.strip() for o in objections if o.strip()]
#
#     resolved_objections = (inputs.get("resolved_objections", AIMessage(content="")).content.split(",")
#                            if inputs.get("resolved_objections").content else [])
#     resolved_objections = [r.strip() for r in resolved_objections if r.strip()]
#
#     tariffs = (inputs.get("tariffs", AIMessage(content="")).content.split(",")
#                if inputs.get("tariffs").content else [])
#     tariffs = [t.strip() for t in tariffs if t.strip()]
#
#     summary_exact = (
#         f"# 1. Выявлены Потребности: {', '.join(needs) if needs else 'потребностей не обнаружено'}\n"
#         f"# 2. Рассказанные Преимущества: {', '.join(benefits) if benefits else 'преимущества не были рассказаны'}\n"
#         f"# 3. Возражения клиента: {', '.join(objections) if objections else 'возражений не обнаружено'}\n"
#         f"# 4. Возражения клиента отработаны: {', '.join(resolved_objections) if resolved_objections else 'отработки не обнаружено'}\n"
#         f"# 5. Конкретика - оговоренная конкретика - курсы, цены: {', '.join(tariffs) if tariffs else 'не обнаружено'}\n"
#     )
#
#     return {**inputs, "summary_exact": summary_exact}


logger = ChainLogger(prefix="[Отчет экстракторов]", debug_mode=True)

def extract_list(inputs: Dict, key: str) -> list:
    """Безопасно извлекает список элементов из AIMessage, разделяя запятыми."""
    msg = inputs.get(key)
    if msg and isinstance(msg, AIMessage) and msg.content:
        return [x.strip() for x in msg.content.split(",") if x.strip()]
    return []

def create_summary_exact(inputs: Dict, debug_mode: bool = False) -> Dict:
    """Создаёт отчет на основе результатов экстракторов."""
    if debug_mode:
        logger.log("init", "info", f"Summary inputs: {list(inputs.keys())}")

    needs = extract_list(inputs, "needs")
    benefits = extract_list(inputs, "benefits")
    objections = extract_list(inputs, "objections")
    resolved_objections = extract_list(inputs, "resolved_objections")
    tariffs = extract_list(inputs, "tariffs")

    summary_exact = (
        f"# 1. Выявлены Потребности: {', '.join(needs) if needs else 'потребностей не обнаружено'}\n"
        f"# 2. Рассказанные Преимущества: {', '.join(benefits) if benefits else 'преимущества не были рассказаны'}\n"
        f"# 3. Возражения клиента: {', '.join(objections) if objections else 'возражений не обнаружено'}\n"
        f"# 4. Возражения клиента отработаны: {', '.join(resolved_objections) if resolved_objections else 'отработки не обнаружено'}\n"
        f"# 5. Конкретика - оговоренная конкретика - курсы, цены: {', '.join(tariffs) if tariffs else 'не обнаружено'}\n"
    )

    if debug_mode:
        logger.log("result", "info", f"Generated summary_exact:\n{summary_exact}")

    return {**inputs, "summary_exact": summary_exact}