from aiogram.types import KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


async def reply_start_keyboard(items: list, buttons_per_row: int = 2):
    kb = ([[InlineKeyboardButton(text=f"{item.get('name')}",
                                 callback_data=f"{item.get('callback_data')}") for item in line]
           for line in (items[i: i + buttons_per_row]
                        for i in range(0, len(items), buttons_per_row))])
    return InlineKeyboardMarkup(
        inline_keyboard=kb,
    )


async def reply_inline_keyboard(items: list, buttons_per_row: int = 1):
    kb = ([[KeyboardButton(text=item) for item in line]
           for line in (items[i: i + buttons_per_row]
                        for i in range(0, len(items), buttons_per_row))])
    return ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Выберите действие"
    )


async def get_inline_keyboard(buttons_data: list, btn_per_row: int = 2):
    """
    buttons
    :param btn_per_row: количество кнопок в строке
    :param buttons_data: = [("📝 Изменить имя", "change_name"),
               ("🎚️ Изменить скорость звука", "change_speed"),
               ("🌡️ Изменить температуру", "change_temperature"),
               ("🎧 Аудио ответы в чатах", "change_audio_responses")]
    :return:
    """
    builder = InlineKeyboardBuilder()

    for text, callback_data in buttons_data:
        builder.button(text=text, callback_data=callback_data)
    builder.adjust(btn_per_row)
    return builder.as_markup()


# async def get_task_confirm_inline_keyboard():
#     """inline клавиатура подтверждения/отклонения номера задачи"""
#     builder = InlineKeyboardBuilder()
#     buttons_data = [(f"{YES_EMOJI} ДА", "task_number_confirmed"),
#                     (f"{NO_EMOJI} НЕТ", "task_number_unconfirmed"),
#                     ]
#     for text, callback_data in buttons_data:
#         builder.button(text=text, callback_data=callback_data)
#     builder.adjust(2)
#     return builder.as_markup()

async def get_task_confirm_inline_keyboard():
    """Создаёт inline-клавиатуру подтверждения/отклонения номера задачи"""

    kb = [
        [InlineKeyboardButton(text="✅ ДА", callback_data="task_number_confirmed"),
         InlineKeyboardButton(text="❌ НЕТ", callback_data="task_number_unconfirmed")]
    ]

    return InlineKeyboardMarkup(inline_keyboard=kb)
