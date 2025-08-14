import json
from typing import List

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable

from neuro_salesman.config import DEFAULT_LLM_MODEL


def create_router_chain(debug_mode: bool = False):
    worker = EXTRACTOR_ROLES.get("router", {
        "name": "Router",
        "temperature": 0.0,
        "model_name": DEFAULT_LLM_MODEL,
        "system_prompt": "",
        "instructions": ""
    })
    llm = ChatOpenAI(model=worker.get("model_name", DEFAULT_LLM_MODEL), temperature=worker.get("temperature", 0))

    def build_system_prompt(needs: List[str]) -> str:
        base_prompt = worker.get("system_prompt", "")
        if len(needs) > 5:
            base_prompt += '''
                ["Специалист по отработке возражений", "Специалист по презентациям", "Специалист по Zoom", “Специалист по завершению”]. 
                Ты знаешь, за что отвечает каждый специалист:
                    #1 Специалист по отработке возражений: участвует, если клиент высказал возражение или сомнение; клиент чем-то недоволен или не всё устраивает в продукте.
                    #2 Специалист по презентациям: участвует, если клиент выразил заинтересованность курсами, нужно презентовать курс или компанию Академия ДПО (повторно презентовать запрещено).
                    #3 Специалист по Zoom: участвует, если клиент говорит, что курс ему подходит, выражает готовность к покупке, обсуждает время созвона или предоставляет контакты.
                    #4 Специалист по завершению: участвует, если клиент завершает диалог (например, "спасибо", "всё понятно").
            '''
        else:
            base_prompt += '''
                ["Специалист по выявлению потребностей", "Специалист по отработке возражений", "Специалист по презентациям", "Специалист по Zoom", “Специалист по завершению”]. 
                Вот описание специалистов:
                    #1 Специалист по выявлению потребностей: всегда участвует в ответе.
                    #2 Специалист по отработке возражений: участвует, если клиент высказал возражение или сомнение; клиент чем-то недоволен или не всё устраивает в продукте.
                    #3 Специалист по презентациям: участвует, если клиент выразил заинтересованность курсами, нужно презентовать курс или компанию Академия ДПО (повторно презентовать запрещено).
                    #4 Специалист по Zoom: участвует, если клиент говорит, что курс ему подходит, выражает готовность к покупке, обсуждает время созвона или предоставляет контакты.
                    #5 Специалист по завершению: участвует, если клиент завершает диалог (например, "спасибо", "всё понятно").
            '''
        base_prompt += '''
            Твоя задача: определить по сообщению клиента, на основании Точного саммари и Хронологии предыдущих сообщений диалога, каких специалистов из Перечня надо выбрать, чтобы они участвовали в ответе клиенту.
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
        def __init__(self, chain, debug_mode):
            self.chain = chain
            self.debug_mode = debug_mode

        def invoke(self, inputs, config=None, **kwargs):
            needs = inputs.get("needs", AIMessage(content="")).content.split(",") if inputs.get("needs").content else []
            needs = [n.strip() for n in needs if n.strip()]
            system_prompt = build_system_prompt(needs)
            summary_history = "\n".join(inputs.get("histories", []))
            result = self.chain.invoke(
                {
                    "system_prompt": system_prompt,
                    "instructions": worker.get("instructions", ""),
                    "last_message_from_client": inputs.get("last_message_from_client", ""),
                    "summary_history": summary_history,
                    "summary_exact": inputs.get("summary_exact", "")
                },
                config=config,
                **kwargs
            )
            if self.debug_mode:
                print(f"[Router] Input: {inputs}")
                print(f"[Router] Output: {result}")
            try:
                # Парсим результат как JSON
                output_router_list = json.loads(result.content.replace("'", '"'))
            except json.JSONDecodeError:
                output_router_list = [s.strip() for s in result.content.split(",") if s.strip()]
            return {**inputs, "router_output": output_router_list}

    chain = prompt_template | llm
    return RouterRunnable(chain, debug_mode)