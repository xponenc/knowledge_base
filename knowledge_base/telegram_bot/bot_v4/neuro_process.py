import asyncio
import json
import os
import re
import time
from typing import Dict, List

import openai
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from telegram_bot.bot_v4.neuro_config import EXPERTS, SENIOR_CONFIG, STYLIST_CONFIG, EXTRACTOR_ROLES
from telegram_bot.bot_v4.neuro_price import get_price, format_cost

verbose_mode = True

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

vectordb = FAISS.load_local('test_faiss_db_it_courses', OpenAIEmbeddings(),
                            allow_dangerous_deserialization=True)


async def ask_neuro(telegram_id: int,
                    text: str,
                    user_profile: Dict,
                    ):
    has_been_inactive = True  # TODO заглушка

    history_chat = user_profile.get("history_chat", [])
    history_user = user_profile.get("history_user", [])
    history_manager = user_profile.get("history_manager", [])
    neuro_data = user_profile.get("neuro_data", {})
    role = user_profile.get("role")

    if role == "client":
        while True:
            client_question = input(f'Вопрос клиента: ')  # TODO заглушка
            history_user.append(client_question)
            history_chat.append(f"Клиент: {client_question}")
            start_time = time.monotonic()
            if has_been_inactive:
                greetings = get_greetings(client_question)  # запомнили приветствие
            else:
                greetings = ""
            has_been_inactive = False  # TODO Заглушка
            without_hello = get_seller_answer(
                history_user=history_user,
                user_message=client_question,
                history_manager=history_manager,
                history_chat=history_chat,
                neuro_data=neuro_data,
                verbose=verbose_mode
            )
            if len(history_chat) == 1 and 'None' not in greetings:
                main_answer = (f'{greetings} меня зовут Василий, я менеджер отдела продаж в Академиии Дополннительного'
                               f' профессионального образрвания (Академии ДПО). ') + without_hello
            else:
                main_answer = without_hello

            print(f'Василий: {main_answer}')

            history_chat.append(f"Менеджер: {without_hello}")
            history_manager.append(without_hello)

            end_time = time.monotonic()
            print(f"Время, затраченное на итерацию: {end_time - start_time:.2f} секунды")

    else:
        pass
        # employee_question = input(f'{bcolors.BGCYAN}Вопрос сотрудника:{bcolors.ENDC} ')
        # history_chat.append(f"Сотрудник: {employee_question}")
        # answer = answer_employee(system_employee, user_employee, db, employee_question)
        # history_chat.append(f"Менеджер: {answer}")
        # # Запуск сохранения
        # from datetime import datetime
        # text_file = f'dialog_{datetime.now().strftime("%d.%m.%Y_%H.%M.%S")}.txt'
        # print(f'{bcolors.BGGREEN}Василий:{bcolors.ENDC} {wrap(remove_newlines(answer))}')


def get_greetings(
        text: str,
        model: str = "gpt-4.1-nano",
        temperature: float = 0,
):
    """Выявляет приветствие в тексте"""
    system_prompt = '''
      Приветствие - это выражение приветствия или приветственное сообщение,
      которое отправляется или произносится в начале общения с кем-либо.
      Приветствие может быть формальным или неформальным, зависеть от культуры и контекста.
      Оно служит для демонстрации вежливости, дружелюбия и желания установить контакт с собеседником.
      Приветствия могут быть разными в различных языках и культурах, от простого 'привет' или 'здравствуйте'
      до более формальных или традиционных выражений.
      Твоя задача выявить в Тексте клиента Приветствие.
      В свой ответ включи только найденное Приветствие.
      Если в тексте клиента нет Приветствия, верни: ''.
    '''
    user = f'Текст клиента: {text}'
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    completion = openai.chat.completions.create(model=model, messages=messages, temperature=temperature)
    answer = completion.choices[0].message.content
    return answer


def get_seller_answer(history_user,
                      user_message: str,
                      history_manager,
                      history_chat,
                      neuro_data: dict,
                      verbose: bool = False):
    """Ансамбль моделей для формирования ответа нейро-продажника"""

    for msg in history_chat:
        print(msg)

    output_router_list = []

    needs = neuro_data.get("needs", [])
    benefits = neuro_data.get("benefits", [])
    objections = neuro_data.get("objections", [])
    resolved_objections = neuro_data.get("resolved_objections", [])
    tariffs = neuro_data.get("tariffs", [])
    summary = neuro_data.get("summary", "")

    # Выявление ПОТРЕБНОСТЕЙ в вопросе пользователя
    worker = EXTRACTOR_ROLES.get("needs_extractor")
    current_needs = extract_entity_from_statement(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        analysis_text=f"Клиент: {user_message}",
        verbose=verbose_mode,
    )
    if current_needs:
        needs.append(current_needs)
        needs = list_cleaner(needs)

    # Выявление ПРЕИМУЩЕСТВ в ответе менеджера
    worker = EXTRACTOR_ROLES.get("benefits_extractor")
    current_benefits = extract_entity_from_statement(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        analysis_text=history_manager[-1] if history_manager else "",
        verbose=verbose_mode,
    )
    if current_benefits:
        benefits.append(current_benefits)
        benefits = list_cleaner(benefits)

    # Выявление ВОЗРАЖЕНИЙ в сообщение клиента
    worker = EXTRACTOR_ROLES.get("objection_detector")
    current_objection = extract_entity_from_statement(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        analysis_text=f"Клиент: {user_message}",
        verbose=verbose_mode,
    )
    if current_objection:
        objections.append(current_objection)
        objections = list_cleaner(objections)

    # Выявление ОТРАБОТАННЫХ ВОЗРАЖЕНИЙ в ответе менеджера
    worker = EXTRACTOR_ROLES.get("resolved_objection_detector")
    current_resolved_objections = extract_entity_from_statement(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        analysis_text='\n'.join(history_chat[-3:-1]),
        verbose=verbose_mode,
    )

    resolved_objections.append(current_resolved_objections)
    resolved_objections = list_cleaner(resolved_objections)

    # Выявление ТАРИФОВ
    worker = EXTRACTOR_ROLES.get("resolved_objection_detector")
    current_tariff = extract_entity_from_statement(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        analysis_text=history_manager[-1] if history_manager else "",
        verbose=verbose_mode,
    )

    if current_tariff:
        tariffs.append(current_tariff)
        tariffs = list_cleaner(tariffs)

    #6. Выделим ключи из последних сообщений клиента и менеджера (предыдущий вопрос+ответ)
    k = 2 if len(history_user) > 1 else 1
    if history_manager and len(history_manager) > 0:
        manager_list = history_manager[-1:]
    else:
        manager_list = []
    worker = EXTRACTOR_ROLES.get("topic_phrase_extractor")
    topic_phrase = get_topic_phrase_questions(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instruction=worker.get("instructions"),
        user_history=history_user[-k:],
        manager_history=manager_list,
        verbose=verbose_mode
    )
    # try:
    #     general_topic_phrase = topic_phrase_answer.split(",")
    # except ValueError:
    #     general_topic_phrase = topic_phrase_answer

    # 7. Суммаризируем хронологию предыдущих сообщений диалога
    summarized_dialog = summarize_dialog(summary,
                                         history_chat,
                                         verbose=verbose_mode,
                                         )

    # 8. Создаем точное саммари с ключевыми моментами диалога

    summary_exact = f'''
    # 1. Выявлены Потребности: {', '.join(needs) if needs else 'потребностей не обнаружено'}\n
    # 2. Рассказанные Преимущества: {', '.join(benefits) if benefits else 'преимущества не были рассказаны'}\n
    # 3. Возражения клиента: {', '.join(objections) if objections else 'возражений не обнаружено'}\n
    # 4. Возражения клиента отработаны: {', '.join(resolved_objections) if resolved_objections
    else 'отработки не обнаружено'}\n
    # 5. Конкретика - оговоренная конкретика - курсы, цены: {', '.join(tariffs) if tariffs else 'не обнаружено'}\n
    '''

    #  9. Запускаем Диспетчера
    worker = EXTRACTOR_ROLES.get("router")
    output_router = user_question_router(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        question=history_user[-1],
        summary_history=summarized_dialog,
        summary_exact=summary_exact,
        needs_lst=needs,
        verbose=verbose_mode
    )

    # output_router = (output_router.replace("```", '"').replace("python", '')
    #                  .replace("‘", '"').replace("'", '"').strip())
    print(f"{output_router=}")
    try:
        output_router_list = json.loads(output_router)
    except AttributeError:
        pass
    # output_router_list = [s.strip() for s in output_router.split(',')]
    print(f"{output_router_list=}")
    #  10. По списку спецов из ответа Диспетчера запускаем спецов:
    experts_answers = []
    # try:
    #     output_router_fixed = (str(output_router).split(':')[1] + '').replace("‘", '"').replace("'", '"')
    # except:
    #     output_router_fixed = str(output_router).replace("‘", '"').replace("'", '"')
    #
    # try:
    #     output_router_list = json.loads(output_router_fixed)
    # except:
    #     output_router_list = ['Специалист по Zoom', 'Специалист по презентациям']

    try:
        for key_param in output_router_list:
            expert_params = EXPERTS[key_param] | {'question': history_user[-1],
                                                  'summary_history': summarized_dialog,
                                                  'summary_exact': summary_exact,
                                                  'base_topic_phrase': topic_phrase,
                                                  'search_index': vectordb}
            expert_answer = processing_question_by_expert(**expert_params)

            experts_answers.append(f'{expert_params["name"]}: {expert_answer}')

    except AttributeError:
        if verbose:
            print(f'Ответ диспетчера либо не вызывает спецов либо имеет неверный формат:{output_router}')

    # 11. На основании предложения узких спецов запускаем старшего менеджера для подготовки комплексного ответа:

    output_senior = senior_answer(
        name=SENIOR_CONFIG.get("name"),
        system=SENIOR_CONFIG.get("system_prompt"),
        instructions=SENIOR_CONFIG.get("instructions"),
        question=history_user[-1],
        output_spez=experts_answers,
        summary_history=summarized_dialog,
        # base_topicphrase=topic_phrase,
        search_index=vectordb,
        summary_exact=summary_exact,
        temp=SENIOR_CONFIG.get("temperature"),
        verbose=verbose_mode,
        spez_list=output_router_list)

    #12. Запускаем Стилиста:

    stylized_answer = style_response(
        name=STYLIST_CONFIG.get("name"),
        system=STYLIST_CONFIG.get("system_prompt"),
        instructions=STYLIST_CONFIG.get("instructions"),
        answers_content=output_senior,
        temp=STYLIST_CONFIG.get("temperature"),
        model=STYLIST_CONFIG.get("model")
    )

    #13. контрольный выстрел по приветствиям:
    answer_without_greetings = remove_greeting(
        text=stylized_answer,
        verbose=verbose_mode,
    )

    return answer_without_greetings


def extract_entity_from_statement(name: str,
                                  system_prompt: str,
                                  instructions: str,
                                  analysis_text: str,
                                  temp: float = 0,
                                  verbose: bool = False,
                                  model: str = "gpt-4.1-nano"):
    """Функция выявления сущностей из переданного текста анализа"""

    if verbose:
        print(f'\n[{name}]')
        print(f'Анализируемый текст:\n{analysis_text}')

    if not analysis_text.strip():
        if verbose:
            print(f'Недостаточно данных для запуска анализа')
        return ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"{instructions}\n\nТекст для анализа:\n{analysis_text}\n\nОтвет: "}
    ]

    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )

    answer = completion.choices[0].message.content

    if verbose:
        print(completion)
        print(f'{completion.usage.total_tokens} total tokens used (question-answer).')

        print(f'Ответ по {name}: ', answer)
        prompt_token_counter = completion.usage.prompt_tokens
        answer_token_counter = completion.usage.completion_tokens
        total_cost, prompt_cost, answer_cost = get_price(
            prompt_token_counter=prompt_token_counter,
            answer_token_counter=answer_token_counter,
            model_name=model,
        )
        print(f"Итого: {format_cost(total_cost)}$, "
              f"Промпт: {format_cost(prompt_cost)}$, "
              f"Ответ: {format_cost(answer_cost)}$")

    return answer


def list_cleaner(lines: list[str]) -> list[str]:
    """
    Очищает список строк: удаляет пустые, кавычки, разбивает по запятым и \n, удаляет дубликаты и пробелы.
    """
    cleaned = []
    for line in lines:
        line = line.replace('"', '').replace('\n', ',').strip()
        if line:
            parts = line.split(',')
            cleaned.extend(p.strip() for p in parts if p.strip())

    return list(set(cleaned))


def get_topic_phrase_questions(name,
                               user_history,
                               manager_history,
                               system_prompt,
                               instruction,
                               temp=0.0,
                               verbose=False,
                               model="gpt-4.1-nano"):
    """
    ключи из последних сообщений

    эта генерация запускается после каждого нового вопроса пользователя, для выделения ключей в его сообщении,
    также выделяем ключи из последнего ответа менеджера, далее через корректора ключей создадим общий логический
    контекст общения (к накопленному списку ключей добавим новые и скорректируем логику)

    """
    #  в том, что история клиента не пустая мы уверены, проверим, что есть история менеджера
    user_history = '\n'.join(user_history)
    if manager_history:
        text = f'Текст: {user_history}\n\n{manager_history[-1]}'  # TODO Последнее а зачем на входе проверка на срезы
    else:
        text = f'Текст: {user_history}'
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f'''{instruction}
                                    \n\nТекст: {text}
                                    \n\nОтвет: '''}
                ]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,  # Используем более низкую температуру для более определенной суммаризации
    )
    answer = completion.choices[0].message.content
    if verbose:
        print(f'Ответ {name}:\n Ключевые слова для Базы знаний:{answer}')
    return answer


def summarize_dialog(dialog, _history, temp=0.1, verbose=0, model: str = "gpt-4.1-nano"):
    """Саммаризация диалога
        запускается перед работой диспетчера-маршрутизатора для формирования Хронологии предыдущих сообщений диалога.
        Всех специалистов будем просить строить свои ответы логичными относительно этой хронологии"""
    i = 2 if len(
        _history) > 1 else 1  # берем 2 последних сообщения для саммаризации (предыд ответ менеджера и новый вопрос клиента)
    last_statements = ' '.join(_history[-i:])
    messages = [
        {"role": "system", "content": '''
                  Ты супер корректор, умеешь выделять в диалогах все самое важное.
                  Ты знаешь, что при саммаризации нельзя исключать из диалога специальные термины и названия курсов,
                  программ и тарифов.
                  Твоя задача сделать полное саммари на основании Истории предыдущих сообщений диалога и Последних сообщений.
                                         '''},
        {"role": "user", "content": f'''Суммаризируй Диалог, ничего не придумывай от себя. Если клиент представился, сохрани информацию об имени.
                                        Если клиент указал свой номер телефона или адрес электронной почты, обязательно отрази эту информацию в саммари.
                                        Если с клиентом есть договоренность о назначенном дне и времени созвона с менеджером, также отрази это в саммаризации.
                                      \n\nИстория предыдущих сообщений диалога: {dialog}.
                                      \n\nПоследние сообщения: {last_statements}
                                      \n\nОтвет: '''
         }
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,  # Используем более низкую температуру для более определенной суммаризации
    )
    answer = completion.choices[0].message.content
    if verbose:
        print(f'Саммари диалога: {answer}')
    return answer


def user_question_router(name: str,
                         system_prompt: str,
                         instructions: str,
                         question: str,
                         summary_history,
                         summary_exact,
                         temp=0,
                         verbose=0,
                         model="gpt-4.1-nano",
                         needs_lst: List = None
                         ):
    """Диспетчер-маршрутизатор
    модель определяет по контексту, Хронологии предыдущих сообщений диалога и точному саммари каких узких специалистов
     нужно привлечь для подготовки материалов для проактивного ответа Старшего менеджера
    """

    if verbose:
        print('\n==================\n')
        print('[user_question_router]\n')
        print(f'Вопрос клиента:', question)
        print('Саммари диалога:\n==================\n', summary_history)
        print(f'Саммари точное:\n==================\n', summary_exact)

    if needs_lst and len(needs_lst) > 5:
        system_prompt += '''
            ["Специалист по отработке возражений", "Специалист по презентациям", "Специалист по Zoom", “Специалист по завершению”]. 
            Ты знаешь, за что отвечает каждый специалист:
                #1 Специалист по отработке возражений:  этот специалист участвует в ответе клиенту если:
                #1.1 клиент высказал возражение или сомнение;
                #1.2 клиент чем-то недоволен или не все устраивает в продукте;
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
                #4 Специалист по завершению: этот специалист участвует в ответе клиенту в самом конце диалога, его задача 
                отвечать когда пользователь дает понять,
                что завершает диалог и больше не намерен ничего спрашивать, например: "спасибо","все понтяно","хорошо", 
                "ладно" и прочие утвердительные выражения логически завершающие общение.
        '''
    else:
        system_prompt += '''
            ["Специалист по выявлению потребностей", "Специалист по отработке возражений", "Специалист по презентациям", "Специалист по Zoom", 
            “Специалист по завершению”]. 
            Вот описание специалистов:
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
                #5 Специалист по завершению: этот специалист участвует в ответе клиенту в самом конце диалога, его задача 
                отвечать когда пользователь дает понять,
                что завершает диалог и больше не намерен ничего спрашивать, например: "спасибо","все понтяно","хорошо", 
                "ладно" и прочие утвердительные выражения логически завершающие общение.'''
    system_prompt += '''
        Твоя задача: определить по сообщению клиента, на основании твоих знаний, Точного саммари и Хронологии 
        предыдущих сообщений диалога каких специалистов из Перечня надо выбрать, чтобы они участвовали в ответе клиенту.
         Ты всегда очень строго следуешь требованиям к порядку отчета.
     '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'''{instructions}
                                      \n\nВопрос клиента:{question}
                                      \n\nХронология предыдущих сообщений диалога: {summary_history}
                                      \n\nСаммари точное: {summary_exact}
                                      \n\nОтвет: '''}
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    answer = completion.choices[0].message.content
    if verbose:
        print(f'\n==================')
        print(f'{completion.usage.total_tokens} total tokens used (question-answer).')
        print('\n==================\n')
        print(f'Ответ {name}:\n', answer)

    return answer


def processing_question_by_expert(name,
                                  system,
                                  instructions,
                                  question,
                                  summary_history,
                                  summary_exact,
                                  base_topic_phrase: str,
                                  search_index,
                                  temp=0,
                                  verbose=0,
                                  k=5,
                                  model="gpt-4.1-nano"):
    if name in ["Специалист по Zoom", "Специалист по завершению"]:
        docs_content = ''
    else:
        knowledge_base = search_index.similarity_search(base_topic_phrase, k=k)
        docs_content = re.sub(r'\n{2}', ' ', '\n '.join(
            [f'\n==================\n' + doc.page_content + '\n' for doc in knowledge_base]))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f'''{instructions}

         Вопрос клиента:{question}

         Хронология предыдущих сообщений диалога: {summary_history}

         Точное саммари: {summary_exact}

         База Знаний: {docs_content}'''}
    ]
    if verbose:
        print('\n==================\n')
        print(f'Вопрос клиента: ', question)
        print('Саммари диалога:\n==================\n', summary_history)
        print(f'Саммари точное:\n==================\n', summary_exact)
        print(f'База знаний:\n==================\n', docs_content)

    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    answer = completion.choices[0].message.content

    try:
        answer = answer.split(': ')[1] + ' '
    except:
        answer = answer
    answer = answer.lstrip('#3')

    if verbose:
        print(f'\n==================')
        print(f'{completion.usage.total_tokens} total tokens used (question-answer).')
        print('\n==================\n')
        print(f'Ответ {name}:\n {answer}')
    return answer


def senior_answer(name,
                  system,
                  instructions,
                  question,
                  output_spez,
                  summary_history,
                  search_index,
                  summary_exact,
                  temp=0,
                  verbose=0,
                  model="gpt-4.1-nano",
                  spez_list=None
                  ):
    if not spez_list:
        spez_list = []

    # дозаполним system по набору спецов
    if 'Специалист по завершению' in spez_list:
        system += "Специалист по завершению."
    else:
        system += '''
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

    if 'Специалист по Zoom' in spez_list:
        system += '''
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
    system += 'Вы всегда строго следуете порядку отчета'

    # дозаполним instructions по набору спецов
    if 'Специалист по завершению' not in spez_list:
        if 'Специалист по выявлению потребностей' in spez_list:
            instructions += '''
            #5 задача:  Опираясь на свой анализ выберите только один вопрос из ответа Специалист по выявлению потребностей, 
            которого нет в Хронологии предыдущих сообщений диалога и он лучше всего подходит логике Хронологии 
            предыдущих сообщений диалога.
        '''
        else:
            instructions += '''
                #5 задача: Опираясь на свой анализ задайте вопрос, который должен способствовать продолжению диалога, 
                продолжая логику Хронологи предыдущих сообщений диалога.
            '''
        instructions += ''' 
         Не объясняйте свой выбор и ничего не комментируйте, не поясняйте из ответа каких специалистов Вы формируете 
         свой ответ. Порядок отчета: В Вашем ответе должен быть только ответ клиенту (Задача 4) + только вопрос клиенту 
         (Задача 5) (без пояснений и комментариев).
         '''
    else:
        instructions += '''
            В свой ответ только включите ответ Специалист по завершению.
            Порядок отчета: В Вашем ответе должен быть только ответ клиенту.
        '''
    output_spez_content = "\n=====\n".join(output_spez)
    if verbose:
        print(f'Ответы узких специалистов:{output_spez_content}\n')
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f'''{instructions}
                                    \nВопрос клиента: {question}
                                    \nХронология предыдущих сообщений диалога: {summary_history}
                                    \nСаммари точное: {summary_exact}
                                    \nОтветы узких специалистов: {output_spez_content}
                                    \nОтвет: '''}
    ]

    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    answer = completion.choices[0].message.content
    if verbose:
        print(f'Ответ {name}: ', answer)

    return answer


#@title Функция
def style_response(name,
                   system,
                   instructions,
                   answers_content,
                   temp=0,
                   verbose=False,
                   model="gpt-4.1-nano"):
    if verbose:
        print('==================')
        print(f'Текст для стилизации:\n{answers_content}')
    user_assist = f'''
        {instructions}\n\nИсходный текст: Кира, я рад, что ты заинтересовалась нашими курсами. 
        Наши программы обучения позволят тебе погрузиться в мир искусственного интеллекта с самого начала обучения.
        Ты сможешь принять участие в реальных проектах уже с начала обучения, что поможет тебе получить ценный опыт 
        и умения, необходимые для успешной карьеры в этой области.
        Какие области твоей жизни ты бы хотела улучшить с помощью обучения в области искусственного интеллекта?

        Ответ:
    '''
    user_assist2 = f'''
        {instructions}\n\nИсходный текст: У нас в АДО самая обширная база учебного контента по различным 
        специальностям, включая 174 темы, что значительно превосходит количество учебных материалов у конкурентов. 
        Какие возможности для трудоустройства в сфере менеджмента, бухгалтерского дела, педагогики, психологии и т.д
        Вас  интересуют?
        Ответ:
    '''

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_assist},
        {
            "role": "assistant",
            "content": '''Кира, я рад, что Вы заинтересовались нашими курсами. Наши образовательные программы позволят 
            Вам окунуться в мир искусственного интеллекта с самого начала. Участвуя в реальных проектах уже на старте 
            обучения, Вы сможете получить ценный опыт и необходимые умения для успешной карьеры в этой области. 
            Что именно Вы хотели бы улучшить в своей жизни, изучая искусственный интеллект?'''},
        {"role": "user", "content": user_assist2},
        {"role": "assistant", "content": '''У нас в УИИ самая обширная база учебного контента по искусственному 
        интеллекту, включая 174 темы, что значительно превосходит количество учебных материалов у конкурентов, 
        включая SkillBox. Может быть Вас интересуют возможности для трудоустройства в сфере искусственного интеллекта 
        и программирования?'''},
        {"role": "user", "content": f'''{instructions}\n\n
        Исходный текст: {answers_content}\n\nОтвет: '''}
    ]

    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    answer = completion.choices[0].message.content
    if verbose:
        print('\n==================')
        print(f'{completion.usage.total_tokens} total tokens used (question-answer).')
        print('==================')
        print(f'Ответ {name}:\n', answer)

    return answer


def remove_greeting(
        text: str,
        model_name="gpt-4.1-nano",
        model_temperature=0,
        verbose=False
):
    """Очищает текст от приветствий"""
    # system_prompt = '''
    # Ты отличный редактор текстов и лучше всех умеешь находить в тексте приветствие (приветственную фразу).
    # Приветствие - это выражение приветствия или приветственное сообщение,
    # которое отправляется или произносится в начале общения с кем-либо.
    # Приветствие может быть формальным или неформальным, зависеть от культуры и контекста.
    # Оно служит для демонстрации вежливости, дружелюбия и желания установить контакт с собеседником.
    # Приветствия могут быть разными в различных языках и культурах, от простого 'привет' или 'здравствуйте'
    # до более формальных или традиционных выражений.
    # Твоя задача обработать Исходный текст следующим образом: проанализировать Исходный текст и если в нем есть
    # приветствие, то удалить его.
    # Не добавляй никаких комментариев и пояснений. Отвечай только отредактированным текстом.
    # '''

    system_prompt = """
    Ты — высокоточный редактор текста.
    Твоя задача: удалить из начала текста только приветствие (вежливую фразу начала общения).
    Не изменяй остальной текст.
    Не добавляй пояснений, комментариев или новых слов.
    Возвращай только отредактированный текст, без лишнего форматирования.
    """

    # Несколько примеров для повышения точности
    few_shot_examples = [
        {
            "user": "Исходный текст: Добрый день, Кира, я готов рассказать Вам о курсе подробнее. Начнем с тарифов?",
            "assistant": "Кира, я готов рассказать Вам о курсе подробнее. Начнем с тарифов?"
        },
        {
            "user": "Исходный текст: Привет, Сергей! Сегодня у нас отличная погода.",
            "assistant": "Сергей! Сегодня у нас отличная погода."
        },
        {
            "user": "Исходный текст: Здравствуйте, коллеги. Начнем собрание.",
            "assistant": "Коллеги. Начнем собрание."
        }
    ]

    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем few-shot примеры
    for ex in few_shot_examples:
        messages.append({"role": "user", "content": f"{ex['user']}\n\nОтвет:"})
        messages.append({"role": "assistant", "content": ex["assistant"]})

    # Основной запрос
    messages.append({"role": "user", "content": f"Исходный текст: {text}\n\nОтвет:"})

    if verbose:
        print("[Удаление приветствия]")
        print("======================")
        print(f"Текст для обработки:\n{text}")

    completion = openai.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=model_temperature
    )

    answer = completion.choices[0].message.content.strip()

    # fallback, если модель вернула пустой ответ
    if not answer:
        answer = text

    if verbose:
        print("\n======================")
        print(f"{completion.usage.total_tokens} total tokens used.")
        print("======================")
        print(f"Ответ:\n{answer}")

    return answer


def create_test_db():
    # 📄 Примерные документы про IT-курсы
    docs = [
        "Наш курс по Python рассчитан на новичков и длится 8 недель.",
        "Курс по Data Science включает в себя Pandas, NumPy и машинное обучение.",
        "Frontend-разработка на React — это быстрый старт в веб.",
        "Мы обучаем DevOps с нуля, включая Docker и CI/CD.",
        "Backend на Django: API, ORM, миграции, безопасность.",
        "Курс по SQL и работе с базами данных: PostgreSQL, MySQL.",
        "Введение в кибербезопасность: защита систем и анализ уязвимостей.",
        "Курс по Linux для системных администраторов.",
        "Автоматизация задач с помощью Python и Bash.",
        "Основы Git и GitHub для командной работы над проектами.",
    ]

    # 🔧 Преобразуем в формат документов LangChain
    documents = [Document(page_content=doc) for doc in docs]

    # 🧠 Векторизация
    embedding = OpenAIEmbeddings()

    vectordb = FAISS.from_documents(documents, embedding)

    # 💾 Сохраняем локально
    vectordb.save_local("test_faiss_db_it_courses")


if __name__ == "__main__":
    telegram_id = 12345678
    # create_test_db()
    user_profile = {
        "history_chat": [],
        "history_user": [],
        "history_manager": [],
        "neuro_data": {
            "needs": [],
            "benefits": [],
            "objections": [],
            "resolved_objections": [],
            "tariffs": [],
            "summary": [],
        },
        "role": "client"
    }
    asyncio.run(ask_neuro(
        telegram_id=telegram_id,
        user_profile=user_profile,
        text=""
    ))
