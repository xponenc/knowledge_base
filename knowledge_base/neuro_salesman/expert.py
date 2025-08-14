import re
from typing import Dict

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

from neuro_salesman.config import DEFAULT_LLM_MODEL
from neuro_salesman.llm_utils import VerboseLLMChain
from neuro_salesman.roles_config import EXPERTS_ROLES


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
                    "instructions": self.cfg.get("instructions", "")
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
    #
    # class ExpertRunnable(Runnable):
    #     def __init__(self, chain, expert_name, expert_params, debug_mode):
    #         self.chain = chain
    #         self.expert_name = expert_name
    #         self.expert_params = expert_params
    #         self.debug_mode = debug_mode
    #
    #     def invoke(self, inputs, config=None, **kwargs):
    #         if self.expert_name in ["Специалист по Zoom", "Специалист по завершению"]:
    #             docs_content = ""
    #         else:
    #             search_index = self.expert_params.get("search_index")
    #             base_topic_phrase = inputs.get("topic_phrases", AIMessage(content="")).content
    #             knowledge_base = search_index.similarity_search(base_topic_phrase, k=self.expert_params.get("k", 5))
    #             docs_content = re.sub(r'\n{2}', ' ', '\n '.join(
    #                 [f'\n==================\n' + doc.page_content + '\n' for doc in knowledge_base]
    #             ))
    #
    #         if self.debug_mode:
    #             print(f"\n==================\n")
    #             print(f"Вопрос клиента: {inputs.get('last_message_from_client')}")
    #             print(f"Саммари диалога:\n==================\n{inputs.get('summary_history')}")
    #             print(f"Саммари точное:\n==================\n{inputs.get('summary_exact')}")
    #             print(f"База знаний:\n==================\n{docs_content}")
    #
    #         result = self.chain.invoke(
    #             {
    #                 "system": self.expert_params.get("system", ""),
    #                 "instructions": self.expert_params.get("instructions", ""),
    #                 "question": inputs.get("last_message_from_client", ""),
    #                 "summary_history": "\n".join(inputs.get("histories", [])),
    #                 "summary_exact": inputs.get("summary_exact", ""),
    #                 "docs_content": docs_content
    #             },
    #             config=config,
    #             **kwargs
    #         )
    #
    #         answer = result.content
    #         try:
    #             answer = answer.split(": ")[1] + " "
    #         except IndexError:
    #             answer = answer.lstrip("#3")
    #
    #         if self.debug_mode:
    #             print(f"\n==================\n")
    #             print(f"{result.usage_metadata['total_tokens']} total tokens used (question-answer).")
    #             print(f"\n==================\n")
    #             print(f"Ответ {self.expert_name}:\n {answer}")
    #
    #         return f"{self.expert_name}: {answer}"
    #
    # return ExpertRunnable(chain=prompt_template | llm, expert_name=expert_name, expert_params=expert_params,
    #                       debug_mode=debug_mode)


def build_parallel_experts(debug_mode: bool = False):
    chains = {}
    for expert, expert_config in EXPERTS_ROLES.items():
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
