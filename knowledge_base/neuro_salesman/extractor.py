from typing import Dict

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, Runnable

from neuro_salesman.config import DEFAULT_LLM_MODEL
from neuro_salesman.llm_utils import VerboseLLMChain
from neuro_salesman.roles_config import EXTRACTOR_ROLES

#
# def make_bound_chain(chain, cfg):
#     def bound_chain(ctx):
#         text = cfg["text_selector"](ctx)  # выбираем правильный текст из входных данных
#         return chain.invoke({"text": text})
#     return bound_chain
#
#
#
# def make_extractor_chain(
#         chain_name: str,
#         chain_config: Dict,
# ):
#     """
#     chain_config: словарь из EXTRACTOR_ROLES
#     chain_name: ключ для результата
#     """
#
#     model_name = chain_config.get("model_name", DEFAULT_LLM_MODEL)
#     model_temperature = chain_config.get("model_temperature", 0)
#     system_prompt = chain_config.get("system_prompt", "")
#     instructions = chain_config.get("instructions", "")
#     llm = ChatOpenAI(model=model_name, temperature=model_temperature)
#
#     # Объединяем system_prompt + instructions в единый prompt с переменной {text}
#     prompt_template = PromptTemplate(
#         template="{system_prompt}\n{instructions}\n\nТекст для анализа:\n{text}\n\nОтвет:",
#         input_variables=["system_prompt", "instructions", "text"]
#     )
#
#     chain = LLMChain(
#         llm=llm,
#         prompt=prompt_template,
#         output_key=chain_name
#     ).bind(
#         system_prompt=system_prompt,
#         instructions=instructions
#     )
#
#     # chain  = LLMChain(
#     #     llm=llm,
#     #     prompt=prompt_template,
#     #     output_key=chain_name
#     # ).bind(system_prompt=system_prompt)
#
#     return make_bound_chain(chain, chain_config)


def make_bound_chain(chain, cfg, debug_mode=False):
    """
    Оборачивает цепочку, чтобы она выбирала текст из контекста.
    """
    class BoundChainRunnable(Runnable):
        def __init__(self, chain, cfg, debug_mode):
            self.chain = chain
            self.cfg = cfg
            self.debug_mode = debug_mode
            # self.output_key = chain.output_key

        def invoke(self, inputs, config=None, **kwargs):
            text = inputs.get(self.cfg["target"])

            if not text or not text.strip():
                if self.debug_mode:
                    print(f"[{self.cfg.get('verbose_name', 'Extractor')}] Текст пуст — модель не вызывалась.")
                return AIMessage(
                    content="",
                    response_metadata={
                        'token_usage': {
                            'completion_tokens': 0,
                            'prompt_tokens': 0,
                            'total_tokens': 0
                        },
                        'model_name': 'n/a',
                        'finish_reason': 'stop'
                    },
                    usage_metadata={
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0
                    }
                )
            result = self.chain.invoke(
                {
                    "text": text,
                    "system_prompt": self.cfg.get("system_prompt", ""),
                    "instructions": self.cfg.get("instructions", "")
                },
                config=config,
                **kwargs
            )
            return result

    return BoundChainRunnable(chain, cfg, debug_mode)


def make_extractor_chain(chain_name: str, chain_config: Dict, debug_mode=False):
    """
    Создаёт экстрактор на основе конфигурации.
    Использует современный подход LangChain с prompt | llm.
    """

    model_name = chain_config.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = chain_config.get("model_temperature", 0)
    system_prompt = chain_config.get("system_prompt", "")
    instructions = chain_config.get("instructions", "")

    llm = ChatOpenAI(model=model_name, temperature=model_temperature)

    # Промпт с подстановкой system_prompt и instructions
    # prompt_template = PromptTemplate(
    #     template="{system_prompt}\n{instructions}\n\nТекст для анализа:\n{text}\n\nОтвет:",
    #     input_variables=["system_prompt", "instructions", "text"]
    # )

    # Use ChatPromptTemplate for proper message handling
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system_prompt}\n{instructions}"),
        HumanMessagePromptTemplate.from_template("Текст для анализа:\n{text}\n\nОтвет:")
    ])

    # Современная композиция вместо LLMChain
    chain = prompt_template | llm

    # Создаём обёртку для форматирования результата
    # class KeyedRunnable(Runnable):
    #     def __init__(self, chain, output_key):
    #         self.chain = chain
    #         self.output_key = output_key
    #
    #     def invoke(self, inputs, config=None, **kwargs):
    #         result = self.chain.invoke(inputs, config=config, **kwargs)
    #         if debug_mode:
    #             print(f"[Greeting Extractor] Input: {inputs}")
    #             print(f"[Greeting Extractor] Output: {{{self.output_key}: {result}}}")
    #         return {self.output_key: result}

    # Оборачиваем цепочку для добавления output_key
    # keyed_chain = KeyedRunnable(chain, chain_name)

    # Возвращаем обёрнутую цепочку
    # return make_bound_chain(keyed_chain, chain_config, debug_mode=debug_mode)
    return make_bound_chain(chain, chain_config, debug_mode=debug_mode)


def build_parallel_extractors(db_name: str, debug_mode: bool = False):
    chains = {}
    for extractor, extractor_config in EXTRACTOR_ROLES.items():
        chain_name = extractor_config.get("verbose_name", "Extractor")
        chain = make_extractor_chain(chain_name=extractor, chain_config=extractor_config, debug_mode=debug_mode)
        verbose_chain = VerboseLLMChain(chain, chain_name=chain_name, debug_mode=debug_mode)
        chains[extractor] = verbose_chain
    return RunnableParallel(**chains)