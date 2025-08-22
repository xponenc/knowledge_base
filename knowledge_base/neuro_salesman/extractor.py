import time
from typing import Dict

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, Runnable, RunnablePassthrough

from neuro_salesman.chains.chain_logger import ChainLogger
from neuro_salesman.chains.keyed_runnable import KeyedRunnable
from neuro_salesman.config import DEFAULT_LLM_MODEL, EMPTY_MESSAGE, LLM_MAX_RETRIES, LLM_TIMEOUT
from neuro_salesman.llm_utils import VerboseLLMChain
from neuro_salesman.roles_config import NEURO_SALER

#
# def make_bound_chain(chain, cfg, debug_mode=False):
#     """
#     Оборачивает цепочку, чтобы она выбирала текст из контекста.
#     """
#     class BoundChainRunnable(Runnable):
#         def __init__(self, chain, cfg, debug_mode):
#             self.chain = chain
#             self.cfg = cfg
#             self.debug_mode = debug_mode
#             # self.output_key = chain.output_key
#
#         def invoke(self, inputs, config=None, **kwargs):
#             text = inputs.get(self.cfg["target"])
#
#             if not text or not text.strip():
#                 if self.debug_mode:
#                     print(f"[{self.cfg.get('verbose_name', 'Extractor')}] Текст пуст — модель не вызывалась.")
#                 return EMPTY_MESSAGE
#             result = self.chain.invoke(
#                 {
#                     "text": text,
#                     "system_prompt": self.cfg.get("system_prompt", ""),
#                     "instructions": self.cfg.get("instructions", "")
#                 },
#                 config=config,
#                 **kwargs
#             )
#             return result
#
#     return BoundChainRunnable(chain, cfg, debug_mode)


# def make_extractor_chain(chain_config: Dict, debug_mode=False):
#     """
#     Создаёт цепочку-экстрактор на основе конфигурации.
#     """
#
#     model_name = chain_config.get("model_name", DEFAULT_LLM_MODEL)
#     model_temperature = chain_config.get("model_temperature", 0)
#     system_prompt = chain_config.get("system_prompt", "")
#     instructions = chain_config.get("instructions", "")
#
#     llm = ChatOpenAI(model=model_name, temperature=model_temperature)
#
#     # Промпт с подстановкой system_prompt и instructions
#     # prompt_template = PromptTemplate(
#     #     template="{system_prompt}\n{instructions}\n\nТекст для анализа:\n{text}\n\nОтвет:",
#     #     input_variables=["system_prompt", "instructions", "text"]
#     # )
#
#     # Use ChatPromptTemplate for proper message handling
#     prompt_template = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template("{system_prompt}\n{instructions}"),
#         HumanMessagePromptTemplate.from_template("Текст для анализа:\n{text}\n\nОтвет:")
#     ])
#
#     # Современная композиция вместо LLMChain
#     chain = prompt_template | llm
#
#     return make_bound_chain(chain, chain_config, debug_mode=debug_mode)


# def build_parallel_extractors(db_name: str, debug_mode: bool = False):
#     chains = {}
#     for extractor, extractor_config in EXTRACTOR_ROLES.items():
#         chain_name = extractor_config.get("verbose_name", "Extractor")
#         chain = make_extractor_chain(chain_name=extractor, chain_config=extractor_config, debug_mode=debug_mode)
#         verbose_chain = VerboseLLMChain(chain, chain_name=chain_name, debug_mode=debug_mode)
#         chains[extractor] = verbose_chain
#     chains['original_inputs'] = RunnablePassthrough()
#     return RunnableParallel(**chains)


def make_extractor_chain(
        chain_config: Dict,
        debug_mode=False):
    """
    Создаёт цепочку-экстрактор на основе конфигурации.

    Пошагово:
    1. Создает LLM-модель с параметрами из конфигурации.
    2. Формирует промпт из system_prompt и instructions.
    3. Оборачивает цепочку в BoundChainRunnable, которая:
        - выбирает текст из inputs[target];
        - вызывает модель;
        - возвращает результат под ключом output_key в inputs;
        - логирует вход и выход через ChainLogger в зависимости от debug_mode.

    Args:
        chain_name (str): Имя цепочки для логирования.
        chain_config (Dict): Конфигурация цепочки (model_name, instructions, target и др.).
        debug_mode (bool): Включает подробный вывод в консоль и debug-лог.

    Returns:
        KeyedRunnable: объект Runnable, который при вызове invoke возвращает inputs с добавленным output_key.
    """
    model_name = chain_config.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = chain_config.get("model_temperature", 0)
    system_prompt = chain_config.get("system_prompt", "")
    instructions = chain_config.get("instructions", "")
    verbose_name = chain_config.get("verbose_name", "Extractor")

    logger = ChainLogger(prefix=f"[{verbose_name}]", debug_mode=debug_mode)
    logger.log("init", "info", f"Chain started")

    llm = ChatOpenAI(model=model_name,
                     temperature=model_temperature,
                     max_retries=LLM_MAX_RETRIES,
                     timeout=LLM_TIMEOUT
                     )

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system_prompt}\n{instructions}"),
        HumanMessagePromptTemplate.from_template("Текст для анализа:\n{text}\n\nОтвет:")
    ])

    logger.log("init",
               "info",
               f"Chain creation started (model={model_name}, temperature={model_temperature})")

    chain = prompt_template | llm

    target_key = chain_config.get("target", "")

    class BoundChainRunnable(KeyedRunnable):
        def invoke(self, inputs, config=None, **kwargs):
            start_time = time.monotonic()
            text = inputs.get(target_key, "")
            session_info = f"{inputs.get('session_type', 'n/a')}:{inputs.get('session_id', 'n/a')}"
            if not text or not text.strip():
                self.logger.log(session_info, "warning", "Текст пуст — модель не вызывалась")
                return EMPTY_MESSAGE
            try:
                result = self.chain.invoke(
                    {
                        "text": text,

                     }, config=config, **kwargs)
                elapsed = time.monotonic() - start_time

                self.logger.log(session_info, "debug", f"input: {inputs}")
                self.logger.log(session_info, "info", f"finished in {elapsed:.2f}s")
                self.logger.log(session_info, "debug", f"output: {self.output_key}={result}")
                return result
            except Exception as e:
                self.logger.log(session_info, "error", f"Ошибка: {str(e)}", exc=e)
                return EMPTY_MESSAGE

    return BoundChainRunnable(chain, output_key="", prefix=verbose_name, debug_mode=debug_mode)

def build_parallel_extractors(extractors:dict, debug_mode: bool = False):
    """
    Создает параллельный runnable, который запускает все экстракторы одновременно.

    Пошагово:
    1. Для каждого экстрактора из EXTRACTOR_ROLES создается BoundChainRunnable через make_extractor_chain.
    2. Добавляется RunnablePassthrough под ключ 'original_inputs' для пропуска исходных данных.
    3. Возвращается RunnableParallel с набором цепочек.

    Args:
        debug_mode (bool): Включает подробный вывод в консоль и debug-лог.

    Returns:
        RunnableParallel: параллельный runnable с цепочками-экстракторами и passthrough для оригинальных inputs.
    """
    chains = {}
    for extractor, extractor_config in extractors.items():
        chains[extractor] = make_extractor_chain(extractor_config, debug_mode=debug_mode)
    chains['original_inputs'] = RunnablePassthrough()
    return RunnableParallel(**chains)