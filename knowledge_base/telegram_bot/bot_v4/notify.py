from aiogram.enums import ParseMode

from telegram_bot.core import bot, init_dispatcher


async def send_telegram_msg(msg: str, chat_id: int):
    dp = await init_dispatcher()  # Гарантируем, что Dispatcher инициализирован
    await bot.send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.HTML)
