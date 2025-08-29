"""
Модуль визуализации структуры AI-ассистента.

Содержит функции для:
1) генерации Mermaid-диаграммы (flowchart TD);
2) вывода иерархической структуры блоков в виде дерева.

Использует модели:
- Assistant
- Block
- BlockConnection
"""
from collections import defaultdict
from typing import List

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import inspect

from django.db.models import Prefetch

from app_ai_assistants.models import Assistant, Block, BlockConnection
# from app_ai_assistants.services.chain_builder import create_extractor_chain
from neuro_salesman.context import search_with_retriever
from neuro_salesman.expert import make_expert_chain
from neuro_salesman.extractor import make_extractor_chain
from neuro_salesman.reformulate import make_reformulator_chain
from neuro_salesman.retrivers import ensemble_retriever_search
from neuro_salesman.router import create_router_chain
from neuro_salesman.senior import create_senior_chain
from neuro_salesman.stylist import create_stylist_chain
from neuro_salesman.summary import create_extractors_report, update_session_summary


# ======================== Mermaid ============================
def generate_mermaid_for_assistant(assistant: Assistant) -> str:
    """
    Генерирует Mermaid-диаграмму для заданного ассистента.

    Args:
        assistant (Assistant): экземпляр ассистента.

    Returns:
        str: код mermaid (flowchart TD).
    """
    blocks = assistant.blocks.all()
    connections = BlockConnection.objects.filter(from_block__assistant=assistant)

    lines = ["flowchart TD"]

    # --- Узлы ---
    for block in blocks:
        node_id = f"B{block.id}"
        # Экранируем специальные символы в имени блока
        safe_name = block.name.replace('"', r'\"').replace('\n', ' ').replace('[', r'\[').replace(']', r'\]').replace('{', r'\{').replace('}', r'\}').replace('`', '')
        label = f"{safe_name} ({block.block_type})"
        # print(f"[Debug Mermaid] Processing block: id={block.id}, name={block.name}, safe_name={safe_name}, label={label}")
        lines.append(f'    {node_id}["{label}"]')

    # --- Связи ---
    for conn in connections:
        # print(f"[Debug Mermaid] Connection: from B{conn.from_block_id} to B{conn.to_block_id}, order={conn.order}")
        lines.append(f"    B{conn.from_block_id} -->|{conn.order}| B{conn.to_block_id}")

    # --- Привязка классов ---
    for block in blocks:
        node_id = f"B{block.id}"
        lines.append(f"    class {node_id} {block.block_type};")

    # --- Стили ---
    style_map = {
        "greeting": "fill:#ffd700,stroke:#333,stroke-width:2px",
        "extractor": "fill:#90ee90,stroke:#333,stroke-width:2px",
        "summary": "fill:#add8e6,stroke:#333,stroke-width:2px",
        "router": "fill:#ffa07a,stroke:#333,stroke-width:2px",
        "expert": "fill:#f08080,stroke:#333,stroke-width:2px",
        "senior": "fill:#ff6347,stroke:#333,stroke-width:2px",
        "stylist": "fill:#9370db,stroke:#333,stroke-width:2px",
        "remove_greeting": "fill:#d3d3d3,stroke:#333,stroke-width:2px",
        "sequence": "fill:#ffe4b5,stroke:#333,stroke-width:2px",
        "parallel": "fill:#afeeee,stroke:#333,stroke-width:2px",
        "reference": "fill:#d8bfd8,stroke:#333,stroke-width:2px",
        "retriever": "fill:#b0e0e6,stroke:#333,stroke-width:2px",
        "passthrough": "fill:#f5f5f5,stroke:#333,stroke-width:2px"
    }
    for block_type, style in style_map.items():
        lines.append(f"    classDef {block_type} {style}")

    mermaid_code = "\n".join(lines)
    # print(f"[Debug Mermaid] Generated code:\n{mermaid_code}")
    return mermaid_code


# ======================== Текстовое дерево ============================
def _print_block_tree(block: Block, prefix: str = "") -> List[str]:
    """
    Рекурсивно обходит дерево блоков начиная с `block`.

    Args:
        block (Block): текущий блок.
        prefix (str): префикс для визуализации иерархии.

    Returns:
        List[str]: список строк с описанием блоков.
    """
    lines = [f"{prefix}- {block.name} ({block.block_type})"]

    children = BlockConnection.objects.filter(from_block=block).order_by("order")
    for conn in children:
        lines.extend(_print_block_tree(conn.to_block, prefix + "  "))

    return lines


def generate_cytoscape_data(assistant: Assistant) -> dict:
    """
    Генерирует структуру данных для Cytoscape.js на основе блоков ассистента.

    Args:
        assistant (Assistant): экземпляр ассистента.

    Returns:
        dict: JSON-структура с nodes и edges.
    """
    nodes = []
    edges = []

    # Узлы
    for block in assistant.blocks.all():
        nodes.append({
            "data": {
                "id": f"B{block.id}",
                "label": f"{block.name} ({block.block_type})",
                "block_type": block.block_type,
            }
        })

    # Связи
    for conn in BlockConnection.objects.filter(from_block__assistant=assistant):
        edges.append({
            "data": {
                "id": f"E{conn.id}",
                "source": f"B{conn.from_block_id}",
                "target": f"B{conn.to_block_id}",
                "order": conn.order,
            }
        })

    return {"nodes": nodes, "edges": edges}


def build_assistant_structure(assistant: Assistant) -> list[dict]:
    """
    Строит иерархическую структуру ассистента:
    - дети контейнеров идут в children
    - верхнеуровневая цепочка сохраняется как последовательность
    """

    # 1. Подтягиваем блоки и соединения одним запросом
    blocks = {
        b.id: b
        for b in assistant.blocks.all().prefetch_related(
            Prefetch("outgoing_connections", queryset=BlockConnection.objects.all())
        )
    }

    # 2. Разделяем карты для детей и цепочки
    children_map = defaultdict(list)
    sequence_map = defaultdict(list)

    for b in blocks.values():
        for conn in b.outgoing_connections.all():
            if conn.is_child:
                children_map[b.id].append(conn.to_block_id)
            else:
                sequence_map[b.id].append(conn.to_block_id)

    # 3. Рекурсивная функция сборки
    def build_tree(block_id):
        block = blocks[block_id]
        return {
            "id": block.id,
            "name": block.name,
            "block_type": block.block_type,
            "config": block.config,
            "function_source": get_function_source(block.block_type),
            # только иерархические дети
            "children": [build_tree(cid) for cid in children_map.get(block_id, [])],
        }

    # 4. Находим стартовые блоки (без входящих)
    start_blocks = [
        b.id for b in blocks.values() if not b.incoming_connections.exists()
    ]

    # 5. Собираем структуру
    result = []
    for start_id in start_blocks:
        cur_id = start_id
        while cur_id:
            node = build_tree(cur_id)
            result.append(node)
            # переходим по цепочке (если есть следующий)
            next_ids = sequence_map.get(cur_id, [])
            cur_id = next_ids[0] if next_ids else None

    return result


BLOCK_FUNCTIONS = {
    "extractor": make_extractor_chain,
    "report": create_extractors_report,  # тут partial можно показать, но для исходника берем функцию
    "router": create_router_chain,
    "expert": make_expert_chain,
    "senior": create_senior_chain,
    "stylist": create_stylist_chain,
    "summary": update_session_summary,
    "passthrough": lambda x: x,
    "retriever": ensemble_retriever_search,
    "reformulator": make_reformulator_chain,
}


def get_function_source(btype: str):
    fn = BLOCK_FUNCTIONS.get(btype)
    if not fn:
        return "# нет обработчика для этого типа блока"
    try:
        src = inspect.getsource(fn)
    except OSError:
        return "# исходник недоступен"

    return highlight(src, PythonLexer(), HtmlFormatter(style="friendly"))