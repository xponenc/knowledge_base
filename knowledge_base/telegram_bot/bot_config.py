import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

try:
    from base_nodes.local_settings import DEBUG
except ImportError:
    try:
        from base_nodes.product_settings import DEBUG
    except ImportError:
        DEBUG = False

# Реквизиты бота
bot_name = 'DPO Assistant'

# DPO_Assistant_bot
bot_credentials = os.getenv('DPO_BOT_CREDENTIALS')

KB_AI_API_URL = "http://127.0.0.1:8001/api"
KB_AI_API_KEY = os.getenv('KB_AI_API_KEY')


# e-mail notification
if DEBUG:
    admins_mail = ['boykov@mail.ru', ]
    info_mail = ['boykov@mail.ru', ]
else:
    admins_mail = ['boykov@mail.ru', ]
    info_mail = ['boykov@mail.ru', ]


# Реквизиты почтового сервера
# host = os.getenv('mail_host')
# port = os.getenv('mail_port')
# mail_user = os.getenv('mail_user')
# mail_passwd = os.getenv('mail_passwd')
# sender_mail = os.getenv('sender_mail')


# """Иконки"""
TASK_EMOJI = u'\U0001F6E0'
MANAGER_EMOJI = u'\U0001F6DF'
CONSULTATION_EMOJI = u'\U0001F4AC'
STUDY_EMOJI = u'\U0001F393'
YES_EMOJI = u'\U00002705'
NO_EMOJI = u'\U0000274C'
CHECK_EMOJI = u'\U0001F6A9'
QUESTION_EMOJI = u'\U00002754'
EXCLAMATION_EMOJI = u'\U00002757'
RIGHT_ARROW_EMOJI = u'\U000027A1'
LEFT_ARROW_EMOJI = u'\U00002B05'
REPORT_EMOJI = u'\U0001F4C4'
PHOTO_EMOJI = u'\U0001F4F7'
CHANGE_EMOJI = u'\U0001F504'
SALE_EMOJI = u'\U0001F4B0'
STORE_EMOJI = u'\U0001F3F7'
EXCHANGE_EMOJI = u'\U0001F503'
START_EMOJI = u'\U0001F51D'
NODE_EMOJI = u'\U0001F194'
EXPERTISE_EMOJI = u'\U0001F477'
CONTACT_EMOJI = u'\U0001F4CC'
INSTALL_EMOJI = u'\U0001F4E5'
UNINSTALL_EMOJI = u'\U0001F4E4'
SCHEME_EMOJI = u'\U0001F5FA'


MAIN_COMMANDS = {
    "start": {
        "name": f"{START_EMOJI} Старт",
        "help_text": f"Начало работы с ботом",
        "callback_data": "START"
    },
}

TASK_COMMANDS = {
    "ai": {
        "name": f"{CONSULTATION_EMOJI} Консультация",
        "help_text": f"Консультация: любые вопросы по Академии ДПО",
        "callback_data": "AI"
    },
    "info": {
        "name": f"{STUDY_EMOJI} Мои курсы",
        "help_text": f"Информация по вашим курсам и учебным программам",
        "callback_data": "STUDY"
    },
    "tasks": {
        "name": f"{TASK_EMOJI} Мои задания",
        "help_text": f"Информация вашим по заданиям и тестам",
        "callback_data": "TASKS"
    },
    "manager": {
        "name": f"{MANAGER_EMOJI} Помощь",
        "help_text": f"<b>Менеджер</b> - Связаться с менеджером",
        "callback_data": "MANAGER"
    },
}
