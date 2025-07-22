from aiogram import Router, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery

from telegram_bot.bot_config import YES_EMOJI, START_EMOJI
from telegram_bot.bot_v4.start_handlers import MenuStates

manager_router = Router()


@manager_router.message(MenuStates.manager)
async def test_sample(message: Message, state: FSMContext):
    data = await state.get_data()
    last_message = data.get("last_message")

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{YES_EMOJI}\tМенеджер"
        try:
            await message.bot.edit_message_text(text=text, chat_id=message.chat.id,
                                                message_id=message_id, reply_markup=None,
                                                parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    msg = "Уже зову менеджера, будет буквально через полчаса"

    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=f"{START_EMOJI} Главное меню", callback_data="START"),
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


@manager_router.message(Command('manager'))
async def test_sample(message: Message, state: FSMContext):
    await state.set_state(MenuStates.manager)
    data = await state.get_data()
    last_message = data.get("last_message")

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{YES_EMOJI}\tМенеджер"
        try:
            await message.bot.edit_message_text(text=text, chat_id=message.chat.id,
                                                         message_id=message_id, reply_markup=None,
                                                         parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    msg = (
        "Вы находитесь в режиме помощи от менеджера"
    )

    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=f"{START_EMOJI} Главное меню", callback_data="START"),
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


@manager_router.callback_query(F.data == "MANAGER")
@manager_router.callback_query(MenuStates.manager, F.data == "go_back")
async def test_sample_callback(callback: CallbackQuery, state: FSMContext):
    # Закрываем callback-запрос
    await callback.answer()
    await state.set_state(MenuStates.manager)
    data = await state.get_data()
    last_message = data.get("last_message")

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{YES_EMOJI}\tМенеджер"
        try:
            await callback.message.bot.edit_message_text(text=text, chat_id=callback.message.chat.id,
                                                message_id=message_id, reply_markup=None,
                                                parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    msg = (
        "Вы находитесь в режиме помощи от менеджера"
    )

    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=f"{START_EMOJI} Главное меню", callback_data="START"),
        ],
    ])

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
