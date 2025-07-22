import asyncio

from aiogram import Router, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import StateFilter, Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery

from telegram_bot.bot_config import YES_EMOJI
from telegram_bot.bot_v4.api_process import process_message
from telegram_bot.bot_v4.start_handlers import AuthStates, MenuStates
from telegram_bot.core import bot_logger

ai_router = Router()
message_lock = asyncio.Lock()


@ai_router.message(Command('ai'))
async def test_sample(message: Message, state: FSMContext):
    await state.set_state(MenuStates.ai)
    data = await state.get_data()
    last_message = data.get("last_message")

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{YES_EMOJI}\tКонсультация"
        try:
            await message.bot.edit_message_text(text=text, chat_id=message.chat.id,
                                                         message_id=message_id, reply_markup=None,
                                                         parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    msg = (
        "Чем могу помочь?"
    )

    answer_message = await message.answer(
        msg,
        parse_mode=ParseMode.HTML,
    )
    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": None,
    })


@ai_router.callback_query(F.data == "AI")
@ai_router.callback_query(MenuStates.ai, F.data == "go_back")
async def test_sample_callback(callback: CallbackQuery, state: FSMContext):
    # Закрываем callback-запрос
    await callback.answer()
    await state.set_state(MenuStates.ai)
    data = await state.get_data()
    last_message = data.get("last_message")

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{YES_EMOJI}\tКонсультация"
        try:
            await callback.message.bot.edit_message_text(text=text, chat_id=callback.message.chat.id,
                                                message_id=message_id, reply_markup=None,
                                                parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    msg = (
        "Чем могу помочь?"
    )

    answer_message = await callback.message.answer(
        msg,
        parse_mode=ParseMode.HTML,
    )
    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": None,
    })



@ai_router.message(MenuStates.ai)
async def handle_message(message: Message, state: FSMContext):
    telegram_id = message.from_user.id
    async with message_lock:
        response = await process_message(
            telegram_id=telegram_id,
            text=message.text,
            is_user=True,
            incoming_message_id=message.message_id
        )

        answer_message = await message.answer(response["ai_text"], parse_mode=ParseMode.MARKDOWN, )
        await state.update_data(last_message={
            "id": answer_message.message_id,
            "text": response["ai_text"],
        })


