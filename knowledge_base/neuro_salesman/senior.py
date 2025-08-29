from typing import Dict, Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from neuro_salesman.chains.chain_logger import ChainLogger
from neuro_salesman.chains.generic_runnable import GenericRunnable
from neuro_salesman.config import DEFAULT_LLM_MODEL, EMPTY_MESSAGE, LLM_MAX_RETRIES, LLM_TIMEOUT
from neuro_salesman.roles_config import NEURO_SALER
from neuro_salesman.utils import print_dict_structure


def build_system_prompt(experts_list: list, basic_system_prompt: str) -> str:
    if 'Специалист по завершению' in experts_list:
        basic_system_prompt += "Специалист по завершению."
    else:
        basic_system_prompt += '''
            Специалист по отработке возражений, Специалист по презентациям, Специалист по Zoom, Специалист по выявлению потребностей.
            #2 Ваша цель общения: в течение всего диалога выявить потребности клиента, закрыть все возражения клиента 
            и в итоге назначить встречу клиента с экспертом для обсуждения деталей приобретения курса в соответствии 
            с выявленными потребностями.
            Вы всегда строго следуете Инструкциям и порядку отчета.
            #3 Инструкция как отвечать на вопрос клиента:
            ##3.1 При формировании своего ответа вы всегда следуете логике Хронологии предыдущих сообщений диалога и 
            опираетесь на Ответы узких специалистов.
            ##3.2 Презентацию делайте только в том случае, если клиент попросит рассказать о курсе\обучении\академии 
            или она закрывает какие-то потребности, презентуйте опираясь на ответ Специалист по презентациям;
            ##3.3 Если у вас есть ответ Специалист по отработке возражений, то закройте возражения, опираясь на ответ 
            Специалист по отработке возражений;
            ##3.4 Вы знаете, что Вам важно закрыть все возражения клиента;
            ##3.5 В ответе Вам категорически запрещено говорить что Вы выясняете потребности и цели клиента;
            ##3.6 Вам запрещено разговаривать на посторонние темы.
            #4 Инструкция как отвечать на посторонние темы: если в Ответах узких специалистов написано, что вопрос 
            не связан с Академией Дополнительного Профессионального образования, это значит, что нужно вежливо отказаться
            отвечать на вопросы на посторонние темы и уточнить, есть ли у клиента вопросы касающиеся курсов, программ 
            обучения в АДО или самой Академии.
        '''

    if 'Специалист по Zoom' in experts_list:
        basic_system_prompt += '''
            #5 Инструкция как звать клиента на встречу с экспертом в Zoom:
            ##5.1 Проанализируйте ответ специалиста Специалист по Zoom: он должен сообщить Вам текущий этап процесса записи на 
            встречу с экспертом (например, "Этап2").
            ##5.2 Если в ответе Специалист по Zoom нет текущего этапа, то пока назначать встречу с экспертом рано;
            ##5.3 Если в ответе Специалист по Zoom есть текущий этап, найдите в Таблице этапов Инструкцию, соответствующую 
            текущему этапу и подготовьте свой ответ строго в полном соответствии с этой инструкцией.
            Ничего не придумывайте от себя, строго следуйте инструкции текущего этапа:
            ###Таблица этапов:
            |Этап| Инструкция|
            |Этап1| Аргументируйте на основании потребностей клиента зачем ему нужно согласиться на Zoom встречу и 
            задайте вопрос подтверждающий согласие клиента на участие во встрече;|
            |Этап2| Предложите клиенту на выбор три конкретных варианта временных промежутков для встречи 
            (например, "Завтра в 16:00, Завтра в 20:00, Послезавтра в 10:00" и тп) и попросите клиента выбрать 
            (из предложенных) подходящее;|
            |Этап3| Запросите номер телефона и почту и аргументируйте что телефон нужен Вам чтобы отправить 
            ссылку на встречу клиенту;|
            |Этап4| Поблагодарите клиента за приятный диалог и напишите о том что встреча назначена на такой-то 
            день и такое-то время|
            Вы обязаны в точности следовать инструкции текущего этапа записи на встречу, ничего не исключайте из 
            нее и не добавляйте от себя.
        '''
    basic_system_prompt += 'Вы всегда строго следуете порядку отчета'
    return basic_system_prompt


def build_instructions(experts_list: list, basic_instructions: str) -> str:
    if 'Специалист по завершению' not in experts_list:
        if 'Специалист по выявлению потребностей' in experts_list:
            basic_instructions += '''
            #5 задача: Опираясь на свой анализ выберите только один вопрос из ответа Специалист по выявлению потребностей, 
            которого нет в Хронологии предыдущих сообщений диалога и он лучше всего подходит логике Хронологии 
            предыдущих сообщений диалога.
        '''
        else:
            basic_instructions += '''
                #5 задача: Опираясь на свой анализ задайте вопрос, который должен способствовать продолжению диалога, 
                продолжая логику Хронологи предыдущих сообщений диалога.
            '''
        basic_instructions += ''' 
         Не объясняйте свой выбор и ничего не комментируйте, не поясняйте из ответа каких специалистов Вы формируете 
         свой ответ. Порядок отчета: В Вашем ответе должен быть только ответ клиенту (Задача 4) + только вопрос клиенту 
         (Задача 5) (без пояснений и комментариев).
         '''
    else:
        basic_instructions += '''
            В свой ответ только включите ответ Специалист по завершению.
            Порядок отчета: В Вашем ответе должен быть только ответ клиенту.
        '''
    return basic_instructions

#
# def create_senior_chain(debug_mode: bool = False):
#     """
#     Создает цепочку для старшего менеджера в стиле LangChain.
#     """
#     SENIOR_CONFIG = NEURO_SALER.get("SENIOR")
#     model_name = SENIOR_CONFIG.get("model_name", DEFAULT_LLM_MODEL)
#     model_temperature = SENIOR_CONFIG.get("model_temperature", 0)
#     system_prompt = SENIOR_CONFIG.get("system_prompt", "")
#     instructions = SENIOR_CONFIG.get("instructions", "")
#
#     llm = ChatOpenAI(
#         model=model_name,
#         temperature=model_temperature
#     )
#
#     prompt_template = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template("{system_prompt}"),
#         HumanMessagePromptTemplate.from_template(
#             "{instructions}\n\n"
#             "Вопрос клиента: {question}\n\n"
#             "Хронология предыдущих сообщений диалога: {summary_history}\n\n"
#             "Точное саммари: {summary_exact}\n\n"
#             "Ответы узких специалистов: {experts_output}\n\n"
#             "Ответ:"
#         )
#     ])
#
#     class KeyedRunnable(Runnable):
#         def __init__(self, chain, output_key, debug_mode):
#             self.chain = chain
#             self.output_key = output_key
#             self.debug_mode = debug_mode
#
#         def invoke(self, inputs, config=None, **kwargs):
#             # Подготовка входных данных
#             experts_output = [
#                 f"{EXPERTS_ROLES[key].get('verbose_name', 'Expert')}: {value.content}" for key, value in inputs.items()
#                 if isinstance(value, AIMessage) and key in EXPERTS_ROLES
#             ]
#             search_index = inputs.get("search_index", [])
#             if isinstance(search_index, list):
#                 search_index = "\n".join(
#                     [doc.page_content if isinstance(doc, Document) else str(doc) for doc in search_index])
#             summary_history = "\n".join(inputs.get("histories", []))
#             experts_output = "\n=====\n".join(experts_output)
#
#             if self.debug_mode:
#                 print(f"[Debug Senior Chain] inputs: {inputs}")
#                 print(f"[Debug Senior Chain] experts_output: {experts_output}")
#                 print(f"[Debug Senior Chain] search_index: {search_index}")
#                 print(f"[Debug Senior Chain] summary_history: {summary_history}")
#
#             try:
#                 result = self.chain.invoke(
#                     {
#                         "system_prompt": build_system_prompt(inputs),
#                         "instructions": build_instructions(inputs),
#                         "question": inputs.get("last message from client", ""),
#                         "summary_history": summary_history,
#                         "summary_exact": inputs.get("summary_exact", ""),
#                         "experts_output": experts_output
#                     },
#                     config=config,
#                     **kwargs
#                 )
#                 if self.debug_mode:
#                     print(f"[Debug Senior Chain] Output: {{{self.output_key}: {result}}}")
#                 return {**inputs, self.output_key: result.content}
#             except Exception as e:
#                 if self.debug_mode:
#                     print(f"[Debug Senior Chain] Ошибка: {str(e)}")
#                 return {**inputs, self.output_key: f"Ошибка при формировании ответа: {str(e)}"}
#
#     chain = prompt_template | llm
#     return KeyedRunnable(chain, "senior_answer", debug_mode)


def create_senior_chain(
        chain_name: str,
        chain_config: Dict[str, Any],
        session_info: str,
) -> GenericRunnable:
    """
    Создает цепочку для "старшего менеджера" (senior chain) в стиле LangChain.

    SeniorChain агрегирует выводы экспертов, историю диалога и контекст,
    формирует финальный ответ для клиента.

    Args:
        debug_mode (bool):
            Если True — включается отладочный вывод в консоль.

    Returns:
        GenericRunnable:
            Обертка над LLM-цепочкой для роли Senior.
    """

    logger = ChainLogger(prefix=f"{chain_name} (extractor)")
    logger.log(session_info, "info", "Chain started")

    # --- Конфигурация ---
    model_name = chain_config.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = chain_config.get("model_temperature", 0)
    system_prompt = chain_config.get("system_prompt", "")
    instructions = chain_config.get("instructions", "")

    llm = ChatOpenAI(
        model=model_name,
        temperature=model_temperature,
        max_retries=LLM_MAX_RETRIES,
        timeout=LLM_TIMEOUT,
    )

    # --- Prompt-шаблон ---
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system_prompt}"),
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Вопрос клиента: {question}\n\n"
            "Саммари всей переписки: \n\n{current_session_summary}\n\n"
            "Хронология последних сообщений диалога: {last_history}\n\n"
            "Отчет по экстракторам: {extractors_report}\n\n"
            "Ответы узких специалистов:\n\n{experts_output}\n\n"
            "Вопрос клиента: {question}\n\n"
            "Ответ:"
        )
    ])

    chain = prompt_template | llm

    logger.log(
        session_info,
        "info",
        f"Chain creation started (model={model_name}, temperature={model_temperature})"
    )

    # --- input_mapping ---
    def input_mapping(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подготавливает входные данные для senior-цепочки:
        - агрегирует ответы экспертов,
        - собирает историю сообщений,
        - формирует поля для промпта.
        """
        # print("SENIOR ", chain_name)
        # print_dict_structure(inputs)
        # print("\n")

        experts = inputs.get("routers", [])
        experts_output = []
        for expert in experts:
            expert_output = inputs.get(expert, EMPTY_MESSAGE)
            if expert_output.content:
                experts_output.append(f"{expert}: {expert_output.content}")
        experts_output = "\n=====\n".join(experts_output)

        last_history = "\n".join(inputs.get("histories", []))

        return {
            "system_prompt": build_system_prompt(experts_list=experts, basic_system_prompt=system_prompt),
            "instructions": build_instructions(experts_list=experts, basic_instructions=instructions),
            "question": inputs.get("last message from client", ""),
            "last_history": last_history,
            "extractors_report": inputs.get("extractors_report", ""),
            "experts_output": experts_output,
            "current_session_summary": inputs.get("current_session_summary", ""),
        }

    # --- output_mapping ---
    def output_mapping(result: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Объединяет исходные inputs с результатом LLM.
        Результат сохраняется под ключом `chain_name`.
        """
        output = {**inputs, chain_name: result}
        # print("SENIOR ", chain_name)
        # print_dict_structure(output)
        # print("\n")
        return output

    # --- Обертка ---
    return GenericRunnable(
        chain=chain,
        output_key="senior_answer",
        prefix="Senior Chain",
        input_mapping=input_mapping,
        output_mapping=output_mapping,
    )
