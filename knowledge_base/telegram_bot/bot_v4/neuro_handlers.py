import asyncio

from aiogram import Router, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import StateFilter, Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery

from telegram_bot.bot_v4.neuro_process import ask_neuro
from telegram_bot.bot_v4.start_handlers import MenuStates

neuro_router = Router()
message_lock = asyncio.Lock()

@neuro_router.message(MenuStates.ai)
async def handle_message(message: Message, state: FSMContext):
    telegram_id = message.from_user.id
    data = await state.get_data()
    user_profile = data.get("user_data")
    async with message_lock:
        response = await ask_neuro(
            telegram_id=telegram_id,
            user_profile =user_profile,
            text=message.text,
            is_user=True,
            incoming_message_id=message.message_id
        )

        answer_message = await message.answer(response["ai_text"], parse_mode=ParseMode.MARKDOWN, )
        await state.update_data(last_message={
            "id": answer_message.message_id,
            "text": response["ai_text"],
        })


