from aiogram import Router
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from telegram_bot.bot_v4.start_handlers import MenuStates

task_router = Router()


@task_router.message(MenuStates.task)
async def test_sample(message: Message, state: FSMContext):
    msg = "Вы находитесь в тестовом разделе мои задания, тут будут ваши задачи, которые надо выполнить в рамках курса"
    answer_message = await message.answer(msg, parse_mode=ParseMode.HTML, )
    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": None,
    })
