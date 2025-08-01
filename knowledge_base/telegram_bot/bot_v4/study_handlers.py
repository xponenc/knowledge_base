from aiogram import Router, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.filters import Command, StateFilter

from telegram_bot.bot_config import CHECK_EMOJI, YES_EMOJI, START_EMOJI
from telegram_bot.bot_v4.start_handlers import MenuStates

study_router = Router()


@study_router.message(Command('info'))
async def test_sample(message: Message, state: FSMContext):
    await state.set_state(MenuStates.study)
    data = await state.get_data()
    print(data)
    last_message = data.get("last_message")

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{YES_EMOJI}\tМои Курсы"
        try:
            await message.bot.edit_message_text(text=text, chat_id=message.chat.id,
                                                         message_id=message_id, reply_markup=None,
                                                         parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    msg = (
        "<b>🎓 Ваши текущие курсы:</b>\n\n"
        "1. <b>Дизайн и архитектура</b> — создавайте потрясающие проекты!\n"
        "2. <b>Водолазное дело</b> (<i>повышение квалификации</i>) — погрузитесь в профессию!\n\n"
        "<tg-spoiler>📢 Новые курсы стартуют скоро!</tg-spoiler>"
    )
    user_data = data.get("user_data")

    if not user_data:
        msg = ("Вы пока не учитесь в Академии\nно если бы учились тут был бы запрос в БД и список ваших курсов,"
               " а пока ...\n\n") + msg
    else:
        msg = (f"Список курсов для {user_data.get('user_name')}\nтут был бы запрос"
               f" в БД и список ваших курсов,  а пока ...\n\n") + msg

    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🎨 Подробнее о дизайне", callback_data="design_details"),
            InlineKeyboardButton(text="🤿 Подробнее о водолазном деле", callback_data="diving_details"),
        ],
        [
            InlineKeyboardButton(text=f"{START_EMOJI} Главное меню", callback_data="START"),
            InlineKeyboardButton(text="📞 Связаться с нами", callback_data="contact_us"),
        ],
    ])

    answer_message = await message.answer(
        msg,
        parse_mode=ParseMode.HTML,
        reply_markup=answer_keyboard,
    )
    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": answer_keyboard.model_dump_json()
    })


@study_router.callback_query(F.data == "STUDY")
@study_router.callback_query(MenuStates.study, F.data == "go_back")
async def test_sample_callback(callback: CallbackQuery, state: FSMContext):
    # Закрываем callback-запрос
    await callback.answer()
    await state.set_state(MenuStates.study)
    data = await state.get_data()
    last_message = data.get("last_message")

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{YES_EMOJI}\tМои Курсы"
        try:
            await callback.message.bot.edit_message_text(text=text, chat_id=callback.message.chat.id,
                                                         message_id=message_id, reply_markup=None,
                                                         parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    msg = (
        "<b>🎓 Ваши текущие курсы:</b>\n\n"
        "1. <b>Дизайн и архитектура</b> — создавайте потрясающие проекты!\n"
        "2. <b>Водолазное дело</b> (<i>повышение квалификации</i>) — погрузитесь в профессию!\n\n"
        "<tg-spoiler>📢 Новые курсы стартуют скоро!</tg-spoiler>"
    )
    print(f"{data=}")
    user_data = data.get("user_data")

    if not user_data:
        msg = ("Вы пока не учитесь в Академии\nно если бы учились тут был бы запрос в БД и список ваших курсов,"
               " а пока ...\n\n") + msg
    else:
        msg = (f"Список курсов для {user_data.get('user_name')}\nтут был бы запрос"
               f" в БД и список ваших курсов,  а пока ...\n\n") + msg


    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🎨 Подробнее о дизайне", callback_data="design_details"),
            InlineKeyboardButton(text="🤿 Подробнее о водолазном деле", callback_data="diving_details"),
        ],
        [
            InlineKeyboardButton(text=f"{START_EMOJI} Главное меню", callback_data="START"),
            InlineKeyboardButton(text="📞 Связаться с нами", callback_data="contact_us"),
        ],
    ])

    # Отправка сообщения с клавиатурой
    answer_message = await callback.message.answer(
        msg,
        parse_mode=ParseMode.HTML,
        reply_markup=answer_keyboard,
    )
    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": answer_keyboard.model_dump_json()
    })


@study_router.callback_query(MenuStates.study, F.data == "design_details")
async def handle_diving_details(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    data = await state.get_data()
    last_message = data.get("last_message")

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{YES_EMOJI}\tДизайн"
        try:
            await callback.message.bot.edit_message_text(text=text, chat_id=callback.message.chat.id,
                                                         message_id=message_id, reply_markup=None,
                                                         parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⬅️ Вернуться назад", callback_data="go_back"),
            InlineKeyboardButton(text=f"{START_EMOJI} Главное меню", callback_data="START"),
        ],
    ])

    msg = "<b>🎨 Супер-курс для дизайнеров интерьеров</b>\n\nПока это заглушка. Подробности о курсе в разработке!"
    answer_message = await callback.message.answer(
        msg,
        parse_mode=ParseMode.HTML,
        reply_markup=answer_keyboard,
    )
    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": answer_keyboard.model_dump_json()
    })


@study_router.callback_query(MenuStates.study, F.data == "diving_details")
async def handle_diving_details(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    data = await state.get_data()
    last_message = data.get("last_message")

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{YES_EMOJI}\tВодолазное дело"
        try:
            await callback.message.bot.edit_message_text(text=text, chat_id=callback.message.chat.id,
                                                         message_id=message_id, reply_markup=None,
                                                         parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⬅️ Вернуться назад", callback_data="go_back"),
            InlineKeyboardButton(text=f"{START_EMOJI} Главное меню", callback_data="START"),
        ],
    ])

    msg = "<b>🤿 Водолазное дело</b>\n\nПока это заглушка. Подробности о курсе в разработке!"
    answer_message = await callback.message.answer(
        msg,
        parse_mode=ParseMode.HTML,
        reply_markup=answer_keyboard,
    )
    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": answer_keyboard.model_dump_json()
    })


@study_router.callback_query(F.data == "contact_us")
async def handle_contact_us(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    data = await state.get_data()
    last_message = data.get("last_message")

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{YES_EMOJI}\tСвязаться с нами"
        try:
            await callback.message.bot.edit_message_text(text=text, chat_id=callback.message.chat.id,
                                                         message_id=message_id, reply_markup=None,
                                                         parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⬅️ Вернуться назад", callback_data="go_back"),
            InlineKeyboardButton(text=f"{START_EMOJI} Главное меню", callback_data="START"),
        ],
    ])

    msg = "<b>📞 Связаться с нами</b>\n\nПока это заглушка. Скоро добавим контакты!"
    answer_message = await callback.message.answer(
        msg,
        parse_mode=ParseMode.HTML,
        reply_markup=answer_keyboard,
    )
    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": answer_keyboard.model_dump_json()
    })
