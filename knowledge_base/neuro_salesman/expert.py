import re
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

from neuro_salesman.chains.chain_logger import ChainLogger
from neuro_salesman.chains.generic_runnable import GenericRunnable
from neuro_salesman.config import DEFAULT_LLM_MODEL
from neuro_salesman.llm_utils import VerboseLLMChain
from neuro_salesman.roles_config import NEURO_SALER
from neuro_salesman.utils import print_dict_structure


def make_bound_chain(chain, cfg, debug_mode=False):
    """
    Оборачивает цепочку, чтобы для подгрузки контекста поиска из векторной базы.
    """
    class BoundChainRunnable(Runnable):
        def __init__(self, chain, cfg, debug_mode):
            self.chain = chain
            self.cfg = cfg
            self.debug_mode = debug_mode
            # self.output_key = chain.output_key

        def invoke(self, inputs, config=None, **kwargs):
            context_search = self.cfg["context_search"]
            if context_search:
                docs_content = inputs.get("search_index")
            else:
                docs_content = None

            result = self.chain.invoke(
                {
                    "docs_content": f"База Знаний: \n{docs_content}" if context_search else "",
                    "system_prompt": self.cfg.get("system_prompt", ""),
                    "instructions": self.cfg.get("instructions", ""),
                    "last message from client": inputs.get("last message from client", ""),
                    "summary_history": "\n".join(inputs.get("histories", [])),
                    "summary_exact": inputs.get("summary_exact", "")
                },
                config=config,
                **kwargs
            )
            return result

    return BoundChainRunnable(chain, cfg, debug_mode)


def make_expert_chain(chain_name: str, expert_config: Dict, debug_mode: bool = False):
    """
        Создаёт цепочку-эксперта на основе конфигурации.
    """
    model_name = expert_config.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = expert_config.get("model_temperature", 0)
    system_prompt = expert_config.get("system_prompt", "")
    instructions = expert_config.get("instructions", "")

    llm = ChatOpenAI(model=model_name, temperature=model_temperature)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system_prompt}"),
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Вопрос клиента: {last message from client}\n\n"
            "Хронология предыдущих сообщений диалога: {summary_history}\n\n"
            "Точное саммари: {summary_exact}\n\n"
            "База знаний: {docs_content}\n\n"
            "Ответ:"
        )
    ])

    chain = prompt_template | llm

    return make_bound_chain(chain, expert_config, debug_mode=debug_mode)


def create_expert_chain(
        chain_name: str,
        chain_config: Dict[str, Any],
        session_info: str
    ) -> Runnable:
    """
    Создаёт цепочку-эксперта на основе конфигурации и оборачивает её в GenericRunnable.

    Args:
        chain_name (str): Название цепочки-эксперта.
        expert_config (Dict): Конфигурация эксперта.
            - model_name (str): имя модели LLM (по умолчанию DEFAULT_LLM_MODEL).
            - model_temperature (float): температура модели (по умолчанию 0).
            - system_prompt (str): системный промпт для LLM.
            - instructions (str): инструкции для эксперта.
            - context_search (bool): использовать ли поиск в базе знаний.
        debug_mode (bool, optional): Режим отладки (по умолчанию False).

    Returns:
        Runnable: Готовая цепочка-эксперт, которая принимает словарь входных данных и
                  возвращает результат вызова модели.
    """
    # Конфигурация модели
    model_name = chain_config.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = chain_config.get("model_temperature", 0)

    # --- Логгер для отладки и мониторинга ---
    logger = ChainLogger(prefix=f"{chain_name} (expert)")
    logger.log(session_info, "info", "Chain started")

    # Инициализация LLM
    llm = ChatOpenAI(model=model_name, temperature=model_temperature)

    # Шаблон промпта
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system_prompt}"),
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Вопрос клиента: {last_message}\n\n"
            "Саммари всей переписки: {current_session_summary}"
            "Хронология последних сообщений диалога: {last_history}\n\n"
            "Отчеты экстракторов: {extractors_report}\n\n"
            "CONTEXT:\n{docs_content}\n\n#Конец базы знаний\n\n"
            "Ответ:"
        )
    ])

    chain = prompt_template | llm

    logger.log(
        session_info,
        "info",
        f"Chain creation started (model={model_name}, temperature={model_temperature})"
    )

    # ---- Обертка через GenericRunnable ----
    def input_mapping(inputs: Dict) -> Dict:
        """
        Преобразует входные данные в формат, который ожидает цепочка.
        """

        context_search = chain_config.get("context_search", False)

        return {
            "docs_content": (
                f"База Знаний: \n{inputs.get('search_index', '')}"
                if context_search else ""
            ),
            "system_prompt": chain_config.get("system_prompt", ""),
            "instructions": chain_config.get("instructions", ""),
            "last_message": inputs.get("last message from client", ""),
            "last_history": "\n".join(inputs.get("histories", [])),
            "extractors_report": inputs.get("extractors_report", ""),
            "current_session_summary": inputs.get("current_session_summary", ""),
        }

    def output_mapping(result, inputs: Dict):
        """
        Оставляем результат LLM без изменений.
        Можно добавить обогащение, если нужно.
        """
        # if chain_name == "Специалист по отработке возражений":
        #     print("EXPERT ", chain_name)
        #     print_dict_structure(inputs)
        #     print("\n")
        return result

    return GenericRunnable(
        chain=chain,
        output_key=chain_name,  # ключ для результата
        prefix=f"ExpertChain[{chain_name}]",  # используется в логах
        input_mapping=input_mapping,
        output_mapping=output_mapping,
    )


def build_parallel_experts(debug_mode: bool = False):
    chains = {}
    experts = NEURO_SALER.get("EXPERTS", {})
    for expert, expert_config in experts.items():
        chain__verbose_name = expert_config.get("verbose_name", "Expert")
        chain = make_expert_chain(chain_name=expert, expert_config=expert_config, debug_mode=debug_mode)
        verbose_chain = VerboseLLMChain(chain, chain_name=chain__verbose_name, debug_mode=debug_mode)
        chains[expert] = verbose_chain
    chains['original_inputs'] = RunnablePassthrough()
    return RunnableParallel(**chains)
#
# def process_experts(inputs: Dict) -> Dict:
#     """Обрабатывает список специалистов, вызывая их цепочки."""
#     router_output = inputs.get("router_output", [])
#     if not router_output:
#         if inputs.get("debug_mode", False):
#             print("[Experts] Ответ диспетчера пуст или некорректен.")
#         return {**inputs, "experts_answers": []}
#
#     experts_answers = []
#     for key_param in router_output:
#         expert_params = EXPERTS_ROLES.get(key_param, {})
#         if not expert_params:
#             if inputs.get("debug_mode", False):
#                 print(f"[Experts] Специалист {key_param} не найден в EXPERTS.")
#             continue
#         expert_params = expert_params | {
#             "question": inputs.get("last_message_from_client", ""),
#             "summary_history": "\n".join(inputs.get("histories", [])),
#             "summary_exact": inputs.get("summary_exact", ""),
#             "base_topic_phrase": inputs.get("topic_phrases", AIMessage(content="")).content,
#             "search_index": inputs.get("search_index")
#         }
#         expert_chain = create_expert_chain(key_param, expert_params, inputs.get("debug_mode", False))
#         answer = expert_chain.invoke(inputs)
#         experts_answers.append(answer)
#
#     return {**inputs, "experts_answers": experts_answers}
