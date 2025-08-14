import json
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable

from neuro_salesman.config import DEFAULT_LLM_MODEL
from neuro_salesman.roles_config import ROUTER_CONFIG


def create_router_chain(debug_mode: bool = False):
    chain_name = ROUTER_CONFIG.get("verbose_name", "Router")
    model_name = ROUTER_CONFIG.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = ROUTER_CONFIG.get("model_temperature", 0)
    instructions = ROUTER_CONFIG.get("instructions", "")

    llm = ChatOpenAI(model=model_name, temperature=model_temperature)

    def build_system_prompt(needs: List[str]) -> str:
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

    class RouterRunnable(Runnable):
        def __init__(self, chain_name, chain, debug_mode):
            self.chain_name = chain_name
            self.chain = chain
            self.debug_mode = debug_mode

        def invoke(self, inputs, config=None, **kwargs):
            needs = inputs.get("needs", AIMessage(content="")).content.split(",") if inputs.get("needs").content else []
            needs = [n.strip() for n in needs if n.strip()]
            system_prompt = build_system_prompt(needs)
            summary_history = "\n".join(inputs.get("histories", []))
            if self.debug_mode:
                print(f"[{self.chain_name}] inputs: {inputs}")
            result = self.chain.invoke(
                {
                    "system_prompt": system_prompt,
                    "instructions": instructions,
                    "last_message_from_client": inputs.get("last_message_from_client", ""),
                    "summary_history": summary_history,
                    "summary_exact": inputs.get("summary_exact", "")
                },
                config=config,
                **kwargs
            )
            if self.debug_mode:
                print(f"[{self.chain_name}] outputs: {result}")
            try:
                # Парсим результат как JSON
                output_router_list = json.loads(result.content.replace("'", '"'))
            except json.JSONDecodeError:
                output_router_list = [s.strip() for s in result.content.split(",") if s.strip()]
            return {**inputs, "router_output": output_router_list}

    chain = prompt_template | llm
    return RouterRunnable(chain_name, chain, debug_mode)

