import json
from typing import Dict

import openai

from telegram_bot.bot_v4.neuro_config import NEURO_ROLES


async def ask_neuro(telegram_id: int,
                    text: str,
                    is_user: bool,
                    user_profile: Dict,
                    incoming_message_id: int):
    has_been_inactive = True  # TODO заглушка
    role = user_profile.get("role")

    if role == "client":
        while True:
            if has_been_inactive:
                greetings = get_greetings(text)  # запомнили приветствие

            without_hello = get_seller_answer(history_user, history_manager, history_chat)
            if len(history_chat) == 1 and 'None' not in hello_word:
                main_answer = f'{hello_word} меня зовут Василий, я менеджер отдела продаж в Академиии Дополннительного профессионального образрвания (Академии ДПО). ' + without_hello
            else:
                main_answer = without_hello

            #   print(f'{bcolors.BGGREEN}Василий:{bcolors.ENDC}\n {insert_newlines(remove_newlines(main_answer), 160)}')
            print(f'{bcolors.BGGREEN}Василий:{bcolors.ENDC} {wrap(remove_newlines(main_answer))}')

            history_chat.append(f"Менеджер: {without_hello}")
            history_manager.append(
                without_hello)  # не будем Василия хранить в истории, чтобы не путать gpt лишними именами

            end_time = time.time()  # Конец итерации
            print(f"Время, затраченное на итерацию: {end_time - start_time:.2f} секунды")
            history_chat.append(f"Клиент: {client_question}")
            if len(history_user) == 1: hello_word = sufler(history_user)  # запомнили приветствие
            if client_question.lower() in ['stop', 'стоп']:
                break
            without_hello = get_seller_answer(history_user, history_manager, history_chat)
            if len(history_chat) == 1 and 'None' not in hello_word:
                main_answer = f'{hello_word} меня зовут Василий, я менеджер отдела продаж в Академии ДПО. ' + without_hello
            else:
                main_answer = without_hello

        # Запуск сохранения
        from datetime import datetime
        text_file = f'dialog_{datetime.now().strftime("%d.%m.%Y_%H.%M.%S")}.txt'
    else:
        employee_question = input(f'{bcolors.BGCYAN}Вопрос сотрудника:{bcolors.ENDC} ')
        history_chat.append(f"Сотрудник: {employee_question}")
        answer = answer_employee(system_employee, user_employee, db, employee_question)
        history_chat.append(f"Менеджер: {answer}")
        # Запуск сохранения
        from datetime import datetime
        text_file = f'dialog_{datetime.now().strftime("%d.%m.%Y_%H.%M.%S")}.txt'
        print(f'{bcolors.BGGREEN}Василий:{bcolors.ENDC} {wrap(remove_newlines(answer))}')


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

    return completion


def get_seller_answer(history_user,
                      history_manager,
                      history_chat,
                      verbose: bool = False):
    """Ансамбль моделей для формирования ответа нейро-продажника"""
    output_router_list = []

    neuro_data = data.get("neuro_data", {})
    needs = neuro_data.get("needs", [])
    benefits = neuro_data.get("benefits", [])
    objections = neuro_data.get("objections", [])
    resolved_objections = neuro_data.get("resolved_objections", [])
    tariffs = neuro_data.get("tariffs", [])
    summary = neuro_data.get("summary", "")

    # Выявление ПОТРЕБНОСТЕЙ в вопросе пользователя
    worker = NEURO_ROLES.get("needs_extractor")
    current_needs = extract_entity_from_statement(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        question=text,
        history=[],
        verbose=verbose,
    )
    if current_needs:
        needs.append(current_needs)
        needs = list_cleaner(needs)

    # Выявление ПРЕИМУЩЕСТВ в ответе менеджера
    worker = NEURO_ROLES.get("benefits_extractor")
    current_benefits = extract_entity_from_statement(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        question='',
        history=history_manager,
    )
    if current_benefits:
        benefits.append(current_benefits)
        benefits = list_cleaner(benefits)

    # Выявление ВОЗРАЖЕНИЙ в сообщение клиента
    worker = NEURO_ROLES.get("objection_detector")
    current_objection = extract_entity_from_statement(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        question=text,
        history='',
    )
    if current_objection:
        objections.append(current_objection)
        objections = list_cleaner(objections)

    # Выявление ОТРАБОТАННЫХ ВОЗРАЖЕНИЙ в ответе менеджера
    worker = NEURO_ROLES.get("resolved_objection_detector")
    current_resolved_objections = extract_entity_from_statement(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        question='',
        history=history_manager,
    )

    resolved_objections.append(current_resolved_objections)
    resolved_objections = list_cleaner(resolved_objections)

    # Выявление ТАРИФОВ
    worker = NEURO_ROLES.get("resolved_objection_detector")
    current_tariff = extract_entity_from_statement(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        question='',
        history=history_manager,
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
    worker = NEURO_ROLES.get("topic_phrase_extractor")
    topic_phrase_completion = get_topic_phrase_questions(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        _user=history_user[-k:],
        _manager=manager_list,
    )
    topic_phrase_answer = topic_phrase_completion.choices[0].message.content

    general_topic_phrase = str(history_user[-1] + ', '
                               + topic_phrase_answer).replace('[', '').replace(']', '').replace(
        "'", '')

    #7. Суммаризируем хронологию предыдущих сообщений диалога
    summarized_comp = summarize_dialog(summary,
                                       history_chat,
                                       temp=0.1,
                                       verbose=verbose_router,
                                       )

    #8. Создаем точное саммари с ключевыми моментами диалога
    tochnoe_summary = f'''
    # 1. Выявлены Потребности: {', '.join(needs) if needs else 'потребностей не обнаружено'}
    # 2. Расказанные Преимущества: {', '.join(benefits) if benefits else 'преимущества не были рассказаны'}
    # 3. Возражения клиента: {', '.join(objections) if objections else 'возражений не обнаружено'}
    # 4. Возражения клиента отработаны: {', '.join(resolved_objections) if resolved_objections else 'отработки не обнаружено'}
    # 5. Конкретика - оговоренная конкретика - курсы, цены: {', '.join(tariffs) if tariffs else 'не обнаружено'}
    '''

    #  9. Запускаем Диспетчера
    worker = NEURO_ROLES.get("router")
    output_router = user_question_router(
        name=worker.get("name"),
        temp=worker.get("temperature"),
        system_prompt=worker.get("system_prompt"),
        instructions=worker.get("instructions"),
        question=history_user[-1],
        summary_history=summarized_dialog,
        summary_exact=tochnoe_summary,
        needs_lst=needs)

    output_router = (output_router.replace("```", '"').replace("python", '')
                     .replace("‘", '"').replace("'", '"').strip())

    #  10. По списку спецов из ответа Диспетчера запускаем спецов:
    output_spez = []
    try:
        output_router_fixed = (str(output_router).split(':')[1] + '').replace("‘", '"').replace("'", '"')
    except:
        output_router_fixed = str(output_router).replace("‘", '"').replace("'", '"')

    try:
        output_router_list = json.loads(output_router_fixed)
    except:
        output_router_list = ['Zoom_Пуш', 'Спец_по_презентациям']

    print(f'{output_router_list=}')
    try:
        for key_param in output_router_list:
            param = spez_config[key_param] | {'question': history_user[-1],
                                              'summary_history': summarized_dialog,
                                              'summary_exact': tochnoe_summary,
                                              'base_topicphrase': general_topic_phrase,
                                              'search_index': vectordb}
            spez_answer = spez_user_question(**param).choices[0].message.content
            try:
                answer = spez_answer.split(': ')[1] + ' '
            except:
                answer = spez_answer
            answer = answer.lstrip('#3')

            output_spez.append(f'{param["name"]}: {wrap(answer)}')

    # if verbose: print(f"\n{bcolors.BGMAGENTA}Ответы спецов:{bcolors.ENDC}\n", '\n\n=========\n'.join(output_spez))
    except:
        if verbose: print(
            f'{bcolors.BGYELLOW}Ответ диспетчера либо не вызывает спецов либо имеет неверный формат:{bcolors.ENDC} {wrap(output_router)}')

    #11. На основании предлоажения узких спецов запускаем страшего менеджера для подготовки комплексного ответа:
    output_senior = senior_answer(
        name=name_senior,
        system=system_prompt_senior,
        instructions=instructions_senior,
        question=history_user[-1],
        output_spez=output_spez,
        summary_history=summarized_dialog,
        base_topicphrase=general_topic_phrase,
        search_index=vectordb,
        summary_exact=tochnoe_summary,
        temp=temperature_senior,
        verbose=verbose_senior,
        k=num_fragments_base_senior,
        model=model_senior,
        spez_list=output_router_list).choices[0].message.content
    #if verbose: print(f"\n{bcolors.BGMAGENTA}senior: {bcolors.ENDC} {wrap(output_senior)}", )
    #12. Запускаем Стилиста:
    output_stilist = stilizator_answer(
        name=name_stilist,
        system=system_prompt_stilist,
        instructions=instructions_stilist,
        answers_content=output_senior,
        temp=temperature_stilist,
        verbose=verbose_stilist,
        model=model_stilist).choices[0].message.content

    #13. контрольный выстрел по приветствиям:
    output_stilist_withouthello = del_hello(
        name=name_stilist,
        system=system_prompt_stilist1,
        instructions=instructions_prompt_stilist1,
        answers_content=output_stilist,
        temp=temperature_stilist,
        verbose=verbose_stilist,
        model=model_stilist).choices[0].message.content

    return output_stilist_withouthello


def extract_entity_from_statement(name: str,
                                  system_prompt: str,
                                  instructions: str,
                                  question: str,
                                  history: list,
                                  temp: float = 0,
                                  verbose: bool = False,
                                  model: str = "gpt-4.1-nano"):
    """Функция выявления сущностей"""
    if verbose:
        print('\n==================\n')
    if verbose and question:
        print(f'Вопрос клиента: {question}')
    if name not in ['Спец по потребностям', 'Спец по возражениям'] and len(
            history):  # эти спецы анализируют только вопрос пользователя
        history_content = history[-1]  # берем только один последний ответ Менеджера в истории
    else:
        history_content = 'сообщений нет'
    if verbose:
        print(f'Предыдущий ответ Менеджера отдела продаж:\n==================\n', history_content)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"{instructions}\n\nВопрос клиента:{question}\n\nПредыдущий ответ Менеджера отдела продаж:"
                    f"\n{history_content}\n\nОтвет: "}
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    answer = completion.choices[0].message.content

    if verbose:
        print('\n==================\n')
        print(f'{completion.usage.total_tokens} total tokens used (question-answer).')
        print('\n==================\n')
        print(f'Ответ {name}:\n', answer)
    # if (name == 'Спец_по_выявлению_потребностей'):
    #     print("Выявлены потребности ")
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


def get_topic_phrase_questions(name, _user, _manager, system_prompt, instructions, temp=0.0, verbose=False,
                               model="gpt-4.1-nano"):
    """
    ключи из последних сообщений

    эта генерация запускается после каждого нового вопроса пользователя, для выделения ключей в его сообщении, также выделяем ключи из последнего ответа
    менеджера, далее через корректора ключей создадим общий логический контекст общения (к накопленному списку ключей добавим новые и скорректируем логику)


    :param name:
    :param _user:
    :param _manager:
    :param system:
    :param instruction:
    :param temp:
    :param verbose:
    :param model:
    :return:
    """
    #  в том, что история клиента не пустая мы уверены, проверим, что есть история менеджера
    join_user = '\n'.join(_user)
    if history_manager:
        text = f'Текст: {join_user}\n\n{history_manager[-1]}'
    else:
        text = f'Текст: {join_user}'
    messages = [{"role": "system", "content": system},
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
        print(f'{bcolors.GREEN}{bcolors.BOLD}Ответ {wrap(name)}:{bcolors.ENDC}\n',
              f'{bcolors.BGYELLOW}Ключевые слова для Базы знаний:{bcolors.ENDC}\n {wrap(answer)}\n=========\n')
    return completion


def summarize_dialog(dialog, _history, temp=0.1, verbose=0, model:str="gpt-4.1-nano"):
    """Саммаризация диалога
        запускается перед работой диспетчера-маршрутизатора для формирования Хронологии предыдущих сообщений диалога.
        Всех специалистов будем просить строить свои ответы логичными относительно этой хронологии"""
    i = 2 if len(_history) > 1 else 1  # берем 2 последних сообщения для саммаризации (предыд ответ менеджера и новый вопрос клиента)
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


def user_question_router(name,
                         system_prompt,
                         instructions,
                         question,
                         summary_history,
                         summary_exact,
                         temp=0,
                         verbose=0,
                         model=MODEL,
                         needs_lst=[]):
    """Диспетчер-маршрутизатор
    модель определяет по контексту, Хронологии предыдущих сообщений диалога и точному саммари каких узких специалистов
     нужно привлечь для подготовки материалов для проактивного ответа Старшего менеджера
    """

    if verbose:
        print('\n==================\n')
        print(f'Вопрос клиента:', question)
        print('Саммари диалога:\n==================\n', summary_history)
        print(f'Саммари точное:\n==================\n',summary_exact)

    if needs_lst and len(needs_lst) > 5:
        system_prompt += '''
            ["Обработчик_возражений", "Спец_по_презентациям", "Zoom_Пуш", “Спец_по_завершению”]. 
            Ты знаешь, за что отвечает каждый специалист:
                #1 Обработчик_возражений:  этот специалист участвует в ответе клиенту если:
                #1.1 клиент высказал возражение или сомнение;
                #1.2 клиент чем-то недоволен или не все устраивает в продукте;
                #2 Спец_по_презентациям: этот специалист участвует в ответе клиенту если клиент выразил 
                заинтересованность курсами, программами и нужно презентовать какой-либо курс представленный в списке 
                для обучения Академии ДПО или какую-то его часть, а также презентовать компанию Академия
                Дополнительного Профессионального Обучения (сокр Академия ДПО), если при этом в Хронологии предыдущих 
                сообщений диалога он это уже презентовал, то повторно презентовать запрещено;
                #3 Zoom_Пуш: этот специалист участвует в ответе клиенту когда:
                #3.1 клиент говорит что курс или программа обучения ему подходит - чтобы позвать клиента на созвон 
                или встречу с экспертом;
                #3.2 клиент выражает готовность к покупке курса или программы обучения - чтобы позвать клиента на 
                созвон или встречу с экспертом для оформления покупки;
                #3.3 клиент обсуждает день и время созвона или встречи с экспертом в Zoom чтобы договориться о встрече;
                #3.4 клиент предоставляет свои контактные данные для отправки приглашения на созвон или встречу в Zoom;
                #4 Спец_по_завершению: этот специалист участвует в ответе клиенту в самом конце диалога, его задача 
                отвечать когда пользователь дает понять,
                что завершает диалог и больше не намерен ничего спрашивать, например: "спасибо","все понтяно","хорошо", 
                "ладно" и прочие утвердительные выражения логически завершающие общение.
        '''
    else:
        system_prompt += '''
            ["Спец_по_выявлению_потребностей", "Обработчик_возражений", "Спец_по_презентациям", "Zoom_Пуш", 
            “Спец_по_завершению”]. 
            Вот описание специалистов:
                #1 Спец_по_выявлению_потребностей: этот специалист всегда участвует в ответе;
                #2 Обработчик_возражений:  этот специалист участвует в ответе клиенту если:
                #2.1 клиент высказал возражение или сомнение;
                #2.2 клиент чем-то недоволен или не все устраивает в продукте;
                #3 Спец_по_презентациям: этот специалист участвует в ответе клиенту если клиент выразил 
                заинтересованность курсами, программами и нужно презентовать курс из предоставленного в программе 
                обучения Академии Дополнительного Профессионального Обучения или какую-то его часть, а также 
                презентовать компанию Академия Дополнительного Профессионального Обучения (сокр Академия ДПО),
                если при этом в Хронологии предыдущих сообщений диалога он это уже презентовал, то повторно 
                презентовать запрещено;
                #4 Zoom_Пуш: этот специалист участвует в ответе клиенту когда:
                #4.1 клиент говорит что курс или программа обучения ему подходит - чтобы позвать клиента на созвон 
                или встречу с экспертом;
                #4.2 клиент выражает готовность к покупке курса или программы обучения - чтобы позвать клиента на 
                созвон или встречу с экспертом для оформления покупки;
                #4.3 клиент обсуждает день и время созвона или встречи с экспертом в Zoom чтобы договориться о встрече;
                #4.4 клиент предоставляет свои контактные данные для отправки приглашения на созвон или встречу в Zoom;
                #5 Спец_по_завершению: этот специалист участвует в ответе клиенту в самом конце диалога, его задача 
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
