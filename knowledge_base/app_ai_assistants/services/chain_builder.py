from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda

from app_ai_assistants.models import Assistant
from neuro_salesman.context import search_with_retriever
from neuro_salesman.router import create_router_chain
from neuro_salesman.senior import create_senior_chain
from neuro_salesman.stylist import create_stylist_chain
from neuro_salesman.summary import create_summary_exact


def build_runnable_from_block(block, debug_mode=False):
    """
    Рекурсивно строит Runnable из блока (и его детей).
    """
    btype = block.block_type

    # --- 1. Листовые узлы ---
    if btype == "extractor":
        return create_extractor_chain(config=block.config, debug_mode=debug_mode)
    elif btype == "summary":
        return RunnableLambda(partial(create_summary_exact, debug_mode=debug_mode))
    elif btype == "router":
        return create_router_chain(debug_mode=debug_mode)
    elif btype == "expert":
        return create_expert_chain(role_config=block.config, debug_mode=debug_mode)
    elif btype == "senior":
        return create_senior_chain(debug_mode=debug_mode)
    elif btype == "stylist":
        return create_stylist_chain(debug_mode=debug_mode)
    elif btype == "passthrough":
        return RunnableLambda(lambda x: x)
    elif btype == "retriever":
        return RunnableLambda(search_with_retriever)

    # --- 2. Контейнеры ---
    elif btype == "parallel":
        children = {child.name: build_runnable_from_block(child, debug_mode) for child in block.children.all()}
        return RunnableParallel(children)
    elif btype == "sequential":
        children = [build_runnable_from_block(child, debug_mode) for child in block.children.all()]
        return RunnableSequence(*children)

    else:
        raise ValueError(f"Неизвестный тип блока: {btype}")


def build_assistant_chain(assistant: Assistant, debug_mode=False):
    """
    Собирает полную цепочку ассистента из top-level блоков.
    """
    top_blocks = (
        assistant.blocks
        .filter(incoming_connections__isnull=True)
        .prefetch_related("children")  # чтобы избежать N+1
        .order_by("id")
    )

    chain_parts = [build_runnable_from_block(b, debug_mode) for b in top_blocks]

    return RunnableSequence(*chain_parts)