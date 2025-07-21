import asyncio

from aiogram import Router, F
from aiogram.enums import ParseMode
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from telegram_bot.bot_v4.api_process import process_message
from telegram_bot.bot_v4.start_handlers import AuthStates, MenuStates
from telegram_bot.core import bot_logger

ai_router = Router()
message_lock = asyncio.Lock()


# @ai_router.message()
# async def debug_any_message(message: Message, state: FSMContext):
#     current = await state.get_state()
#     await message.answer(f"[DEBUG] Состояние: {current}")
#

@ai_router.message(StateFilter(
    # AuthStates.authorized,
    # AuthStates.guest,
    MenuStates.consult,
))
# @ai_router.message(AuthStates.guest)
async def handle_message(message: Message, state: FSMContext):
    telegram_id = message.from_user.id

    # Use lock to ensure messages are processed sequentially
    async with message_lock:
        # Process user message and get AI response
        response = await process_message(
            telegram_id=telegram_id,
            text=message.text,
            is_user=True,
            incoming_message_id=message.message_id
        )

        # Send AI response to user
        answer_message = await message.answer(response["ai_text"], parse_mode=ParseMode.MARKDOWN, )
        await state.update_data(last_message={
            "id": answer_message.message_id,
            "text": response["ai_text"],
        })


