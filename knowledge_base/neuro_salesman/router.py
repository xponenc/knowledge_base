import json
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable

from neuro_salesman.chains.chain_logger import ChainLogger
from neuro_salesman.config import DEFAULT_LLM_MODEL
from neuro_salesman.roles_config import NEURO_SALER

#
# def create_router_chain(debug_mode: bool = False):
#     chain_name = ROUTER_CONFIG.get("verbose_name", "Router")
#     model_name = ROUTER_CONFIG.get("model_name", DEFAULT_LLM_MODEL)
#     model_temperature = ROUTER_CONFIG.get("model_temperature", 0)
#     instructions = ROUTER_CONFIG.get("instructions", "")
#
#     llm = ChatOpenAI(model=model_name, temperature=model_temperature)
#
#     def build_system_prompt(needs: List[str]) -> str:
#         base_prompt = ROUTER_CONFIG.get("system_prompt", "")
#         if len(needs) > 5:
#             base_prompt += '''
#                 ["Специалист по отработке возражений", "Специалист по презентациям", "Специалист по Zoom",
#                  “Специалист по завершению”].
#                 Ты знаешь, за что отвечает каждый специалист:
#                 #1 Специалист по отработке возражений:
#                 #1.1 клиент высказал возражение или сомнение;
#                 #1.2 клиент чем-то недоволен или не всё устраивает в продукте;
#
#                 #2 Специалист по презентациям: этот специалист участвует в ответе клиенту если клиент выразил
#                 заинтересованность курсами, программами и нужно презентовать какой-либо курс представленный в списке
#                 для обучения Академии ДПО или какую-то его часть, а также презентовать компанию Академия
#                 Дополнительного Профессионального Обучения (сокр Академия ДПО), если при этом в Хронологии предыдущих
#                 сообщений диалога он это уже презентовал, то повторно презентовать запрещено;
#
#                 #3 Специалист по Zoom: этот специалист участвует в ответе клиенту когда:
#                 #3.1 клиент говорит что курс или программа обучения ему подходит - чтобы позвать клиента на созвон
#                 или встречу с экспертом;
#                 #3.2 клиент выражает готовность к покупке курса или программы обучения - чтобы позвать клиента на
#                 созвон или встречу с экспертом для оформления покупки;
#                 #3.3 клиент обсуждает день и время созвона или встречи с экспертом в Zoom чтобы договориться о встрече;
#                 #3.4 клиент предоставляет свои контактные данные для отправки приглашения на созвон или встречу в Zoom;
#
#                 #4 Специалист по завершению: этот специалист участвует в ответе клиенту в самом конце диалога,
#                 его задача отвечать когда пользователь дает понять, что завершает диалог и больше не намерен ничего
#                 спрашивать, например: "спасибо","все понятно","хорошо",
#                 "ладно" и прочие утвердительные выражения логически завершающие общение.
#             '''
#         else:
#             base_prompt += '''
#                 ["Специалист по выявлению потребностей", "Специалист по отработке возражений",
#                 "Специалист по презентациям", "Специалист по Zoom", “Специалист по завершению”].
#                 #1 Специалист по выявлению потребностей: этот специалист всегда участвует в ответе;
#                 #2 Специалист по отработке возражений:  этот специалист участвует в ответе клиенту если:
#                 #2.1 клиент высказал возражение или сомнение;
#                 #2.2 клиент чем-то недоволен или не все устраивает в продукте;
#                 #3 Специалист по презентациям: этот специалист участвует в ответе клиенту если клиент выразил
#                 заинтересованность курсами, программами и нужно презентовать курс из предоставленного в программе
#                 обучения Академии Дополнительного Профессионального Обучения или какую-то его часть, а также
#                 презентовать компанию Академия Дополнительного Профессионального Обучения (сокр Академия ДПО),
#                 если при этом в Хронологии предыдущих сообщений диалога он это уже презентовал, то повторно
#                 презентовать запрещено;
#                 #4 Специалист по Zoom: этот специалист участвует в ответе клиенту когда:
#                 #4.1 клиент говорит что курс или программа обучения ему подходит - чтобы позвать клиента на созвон
#                 или встречу с экспертом;
#                 #4.2 клиент выражает готовность к покупке курса или программы обучения - чтобы позвать клиента на
#                 созвон или встречу с экспертом для оформления покупки;
#                 #4.3 клиент обсуждает день и время созвона или встречи с экспертом в Zoom чтобы договориться о встрече;
#                 #4.4 клиент предоставляет свои контактные данные для отправки приглашения на созвон или встречу в Zoom;
#                 #5 Специалист по завершению: этот специалист участвует в ответе клиенту в самом конце диалога,
#                 его задача отвечать когда пользователь дает понять, что завершает диалог и больше не намерен ничего
#                 спрашивать, например: "спасибо","все понятно","хорошо", "ладно" и прочие утвердительные выражения
#                 логически завершающие общение.'''
#         base_prompt += '''
#             Твоя задача: определить по сообщению клиента, на основании Точного саммари и Хронологии предыдущих
#             сообщений диалога, каких специалистов из Перечня надо выбрать, чтобы они участвовали в ответе клиенту.
#             Ты всегда строго следуешь требованиям к порядку отчета.
#         '''
#         return base_prompt
#
#     prompt_template = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template("{system_prompt}"),
#         HumanMessagePromptTemplate.from_template(
#             "{instructions}\n\n"
#             "Вопрос клиента: {last_message_from_client}\n\n"
#             "Хронология предыдущих сообщений диалога: {summary_history}\n\n"
#             "Саммари точное: {summary_exact}\n\n"
#             "Ответ: "
#         )
#     ])
#
#     class RouterRunnable(Runnable):
#         def __init__(self, chain_name, chain, debug_mode):
#             self.chain_name = chain_name
#             self.chain = chain
#             self.debug_mode = debug_mode
#
#         def invoke(self, inputs, config=None, **kwargs):
#             needs = inputs.get("needs", AIMessage(content="")).content.split(",") if inputs.get("needs").content else []
#             needs = [n.strip() for n in needs if n.strip()]
#             system_prompt = build_system_prompt(needs)
#             summary_history = "\n".join(inputs.get("histories", []))
#             if self.debug_mode:
#                 print(f"[{self.chain_name}] inputs: {inputs}")
#             result = self.chain.invoke(
#                 {
#                     "system_prompt": system_prompt,
#                     "instructions": instructions,
#                     "last_message_from_client": inputs.get("last_message_from_client", ""),
#                     "summary_history": summary_history,
#                     "summary_exact": inputs.get("summary_exact", "")
#                 },
#                 config=config,
#                 **kwargs
#             )
#             if self.debug_mode:
#                 print(f"[{self.chain_name}] outputs: {result}")
#             try:
#                 # Парсим результат как JSON
#                 output_router_list = json.loads(result.content.replace("'", '"'))
#             except json.JSONDecodeError:
#                 output_router_list = [s.strip() for s in result.content.split(",") if s.strip()]
#             return {**inputs, "router_output": output_router_list}
#
#     chain = prompt_template | llm
#     return RouterRunnable(chain_name, chain, debug_mode)


"""
Router Chain

Этот модуль отвечает за маршрутизацию диалога между "специалистами"
(выявление потребностей, отработка возражений, презентация, Zoom, завершение).
Он анализирует последнее сообщение клиента, точное саммари и историю диалога,
и возвращает список специалистов, которые должны участвовать в ответе.

Компоненты:
- RouterRunnable: обёртка над LLM-цепочкой, реализующая invoke().
- create_router_chain: фабрика, создающая RouterRunnable с промптом и моделью.
- ChainLogger: для удобного логгирования вызовов цепочки.
"""

# === Логгер цепочки ===
logger = ChainLogger(prefix="[Router]", debug_mode=False)


# === Вспомогательные функции ===
def safe_split(msg: AIMessage) -> List[str]:
    """
    Безопасно разбивает содержимое AIMessage на список строк.

    Args:
        msg (AIMessage): сообщение модели.

    Returns:
        List[str]: список непустых строк.
    """
    if msg and hasattr(msg, "content") and msg.content:
        return [s.strip() for s in msg.content.split(",") if s.strip()]
    return []


def parse_router_output(result_content: str) -> List[str]:
    """
    Парсит вывод LLM-цепочки. Сначала пробует JSON, затем fallback в список строк.

    Args:
        result_content (str): строка от модели.

    Returns:
        List[str]: список специалистов.
    """
    try:
        return json.loads(result_content.replace("'", '"'))
    except json.JSONDecodeError:
        return [s.strip() for s in result_content.split(",") if s.strip()]


def build_system_prompt(needs: List[str]) -> str:
    """
    Собирает system-prompt для LLM в зависимости от количества выявленных потребностей.

    Args:
        needs (List[str]): список выявленных потребностей клиента.

    Returns:
        str: текст system-prompt.
    """
    base_prompt = ROUTER_CONFIG.get("system_prompt", "")
    if len(needs) > 5:
        base_prompt += '''
            ["Специалист по отработке возражений", "Специалист по презентациям", "Специалист по Zoom",
             “Специалист по завершению”]. 
            Ты знаешь, за что отвечает каждый специалист:
            #1 Специалист по отработке возражений: 
            #1.1 клиент высказал возражение или сомнение;
            #1.2 клиент чем-то недоволен или не всё устраивает в продукте;

            #2 Специалист по презентациям: этот специалист участвует в ответе клиенту если клиент выразил 
            заинтересованность курсами, программами и нужно презентовать какой-либо курс представленный в списке 
            для обучения Академии ДПО или какую-то его часть, а также презентовать компанию Академия
            Дополнительного Профессионального Обучения (сокр Академия ДПО), если при этом в Хронологии предыдущих 
            сообщений диалога он это уже презентовал, то повторно презентовать запрещено;

            #3 Специалист по Zoom: этот специалист участвует в ответе клиенту когда:
            #3.1 клиент говорит что курс или программа обучения ему подходит - чтобы позвать клиента на созвон 
            или встречу с экспертом;
            #3.2 клиент выражает готовность к покупке курса или программы обучения - чтобы позвать клиента на 
            созвон или встречу с экспертом для оформления покупки;
            #3.3 клиент обсуждает день и время созвона или встречи с экспертом в Zoom чтобы договориться о встрече;
            #3.4 клиент предоставляет свои контактные данные для отправки приглашения на созвон или встречу в Zoom;

            #4 Специалист по завершению: этот специалист участвует в ответе клиенту в самом конце диалога,
            его задача отвечать когда пользователь дает понять, что завершает диалог и больше не намерен ничего 
            спрашивать, например: "спасибо","все понятно","хорошо", 
            "ладно" и прочие утвердительные выражения логически завершающие общение.
        '''
    else:
        base_prompt += '''
            ["Специалист по выявлению потребностей", "Специалист по отработке возражений", 
            "Специалист по презентациям", "Специалист по Zoom", “Специалист по завершению”]. 
            #1 Специалист по выявлению потребностей: этот специалист всегда участвует в ответе;
            #2 Специалист по отработке возражений:  этот специалист участвует в ответе клиенту если:
            #2.1 клиент высказал возражение или сомнение;
            #2.2 клиент чем-то недоволен или не все устраивает в продукте;
            #3 Специалист по презентациям: этот специалист участвует в ответе клиенту если клиент выразил 
            заинтересованность курсами, программами и нужно презентовать курс из предоставленного в программе 
            обучения Академии Дополнительного Профессионального Обучения или какую-то его часть, а также 
            презентовать компанию Академия Дополнительного Профессионального Обучения (сокр Академия ДПО),
            если при этом в Хронологии предыдущих сообщений диалога он это уже презентовал, то повторно 
            презентовать запрещено;
            #4 Специалист по Zoom: этот специалист участвует в ответе клиенту когда:
            #4.1 клиент говорит что курс или программа обучения ему подходит - чтобы позвать клиента на созвон 
            или встречу с экспертом;
            #4.2 клиент выражает готовность к покупке курса или программы обучения - чтобы позвать клиента на 
            созвон или встречу с экспертом для оформления покупки;
            #4.3 клиент обсуждает день и время созвона или встречи с экспертом в Zoom чтобы договориться о встрече;
            #4.4 клиент предоставляет свои контактные данные для отправки приглашения на созвон или встречу в Zoom;
            #5 Специалист по завершению: этот специалист участвует в ответе клиенту в самом конце диалога, 
            его задача отвечать когда пользователь дает понять, что завершает диалог и больше не намерен ничего 
            спрашивать, например: "спасибо","все понятно","хорошо", "ладно" и прочие утвердительные выражения 
            логически завершающие общение.'''
    base_prompt += '''
        Твоя задача: определить по сообщению клиента, на основании Точного саммари и Хронологии предыдущих 
        сообщений диалога, каких специалистов из Перечня надо выбрать, чтобы они участвовали в ответе клиенту.
        Ты всегда строго следуешь требованиям к порядку отчета.
    '''
    return base_prompt


# === Основной Runnable для Router ===
class RouterRunnable(Runnable):
    """
    Runnable-обёртка для маршрутизатора диалога.

    Отвечает за:
    - подготовку входных данных для LLM (system prompt, инструкции, история, саммари);
    - вызов LLM;
    - парсинг результата;
    - возврат результата в виде inputs + "router_output".
    """

    def __init__(self, chain_name: str, chain, debug_mode: bool = False):
        """
        Args:
            chain_name (str): имя цепочки (для логов).
            chain: связка PromptTemplate | LLM.
            debug_mode (bool): включать ли печать в консоль.
        """
        self.chain_name = chain_name
        self.chain = chain
        self.debug_mode = debug_mode

    def invoke(self, inputs: Dict, config=None, **kwargs) -> Dict:
        """
        Запускает цепочку маршрутизации.

        Args:
            inputs (Dict): словарь входных данных (needs, histories, last_message_from_client, summary_exact).
            config: конфигурация выполнения.
            **kwargs: дополнительные параметры для LLM.

        Returns:
            Dict: inputs + {"router_output": список специалистов}
        """
        needs = safe_split(inputs.get("needs", AIMessage(content="")))
        system_prompt = build_system_prompt(needs)
        summary_history = "\n".join(inputs.get("histories", []))

        llm_inputs = {
            "system_prompt": system_prompt,
            "instructions": ROUTER_CONFIG.get("instructions", ""),
            "last_message_from_client": inputs.get("last_message_from_client", ""),
            "summary_history": summary_history,
            "summary_exact": inputs.get("summary_exact", "")
        }

        logger.log("invoke", "info", f"[{self.chain_name}] inputs={llm_inputs}")

        result = self.chain.invoke(llm_inputs, config=config, **kwargs)

        logger.log("invoke", "info", f"[{self.chain_name}] outputs={result}")

        output_router_list = parse_router_output(result.content)
        return {**inputs,
                "routers": output_router_list,
                "router_output": result}


# === Фабрика цепочки ===
def create_router_chain(router_name: str, debug_mode: bool = False) -> RouterRunnable:
    """
    Фабрика для создания RouterRunnable.

    Args:
        router_name (str): имя роутера
        debug_mode (bool): включить ли debug-режим (печать + логгирование).

    Returns:
        RouterRunnable: готовая цепочка для маршрутизации диалога.
    """
    router_config = NEURO_SALER.get("ROUTERS", {}).get(router_name)
    chain_name = router_config.get("verbose_name", "Router")
    model_name = router_config.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = router_config.get("model_temperature", 0)

    llm = ChatOpenAI(model=model_name, temperature=model_temperature)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system_prompt}"),
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Вопрос клиента: {last_message_from_client}\n\n"
            "Хронология предыдущих сообщений диалога: {summary_history}\n\n"
            "Саммари точное: {summary_exact}\n\n"
            "Ответ: "
        )
    ])

    chain = prompt_template | llm
    return RouterRunnable(chain_name, chain, debug_mode)