from typing import Dict

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from neuro_salesman.config import DEFAULT_LLM_MODEL
from neuro_salesman.roles_config import STYLIST_CONFIG


def create_stylist_chain(debug_mode: bool = False):
    """
    Создает цепочку для стилизации ответа в стиле LangChain.
    """
    model_name = STYLIST_CONFIG.get("model_name", DEFAULT_LLM_MODEL)
    model_temperature = STYLIST_CONFIG.get("model_temperature", 0)
    system_prompt = STYLIST_CONFIG.get("system_prompt", "")
    instructions = STYLIST_CONFIG.get("instructions", "")


    llm = ChatOpenAI(
        model=model_name,
        temperature=model_temperature
    )

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Исходный текст: {answers_content}\n\n"
            "Ответ:"
        ),
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Исходный текст: Кира, я рад, что ты заинтересовалась нашими курсами. "
            "Наши программы обучения позволят тебе погрузиться в мир искусственного интеллекта с самого начала обучения. "
            "Ты сможешь принять участие в реальных проектах уже с начала обучения, что поможет тебе получить ценный опыт "
            "и умения, необходимые для успешной карьеры в этой области. "
            "Какие области твоей жизни ты бы хотела улучшить с помощью обучения в области искусственного интеллекта?\n\n"
            "Ответ:"
        ),
        AIMessage(content='''
            Кира, я рад, что Вы заинтересовались нашими курсами. Наши образовательные программы позволят 
            Вам окунуться в мир искусственного интеллекта с самого начала. Участвуя в реальных проектах уже на старте 
            обучения, Вы сможете получить ценный опыт и необходимые умения для успешной карьеры в этой области. 
            Что именно Вы хотели бы улучшить в своей жизни, изучая искусственный интеллект?
        '''),
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Исходный текст: У нас в АДО самая обширная база учебного контента по различным "
            "специальностям, включая 174 темы, что значительно превосходит количество учебных материалов у конкурентов. "
            "Какие возможности для трудоустройства в сфере менеджмента, бухгалтерского дела, педагогики, психологии и т.д "
            "Вас интересуют?\n\n"
            "Ответ:"
        ),
        AIMessage(content='''
            У нас в УИИ самая обширная база учебного контента по искусственному 
            интеллекту, включая 174 темы, что значительно превосходит количество учебных материалов у конкурентов, 
            включая SkillBox. Может быть Вас интересуют возможности для трудоустройства в сфере искусственного интеллекта 
            и программирования?
        '''),
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Исходный текст: {answers_content}\n\n"
            "Ответ:"
        )
    ])

    class KeyedRunnable(Runnable):
        def __init__(self, chain, output_key, debug_mode):
            self.chain = chain
            self.output_key = output_key
            self.debug_mode = debug_mode

        def invoke(self, inputs, config=None, **kwargs):
            answers_content = inputs.get("senior_answer", "")
            if not answers_content:
                if self.debug_mode:
                    print("[Debug Stylist Chain] Нет текста для стилизации.")
                return {**inputs, self.output_key: ""}

            if self.debug_mode:
                print(f"[Debug Stylist Chain] inputs: {inputs}")
                print(f"[Debug Stylist Chain] answers_content: {answers_content}")

            try:
                result = self.chain.invoke(
                    {
                        "instructions": instructions,
                        "answers_content": answers_content
                    },
                    config=config,
                    **kwargs
                )
                if self.debug_mode:
                    print(f"[Debug Stylist Chain] Output: {{{self.output_key}: {result}}}")
                return {**inputs, self.output_key: result.content}
            except Exception as e:
                # logger.error(f"Ошибка в Stylist Chain: {str(e)}")
                if self.debug_mode:
                    print(f"[Debug Stylist Chain] Ошибка: {str(e)}")
                return {**inputs, self.output_key: f"Ошибка при стилизации ответа: {str(e)}"}

    chain = prompt_template | llm
    return KeyedRunnable(chain, "stylized_answer", debug_mode)