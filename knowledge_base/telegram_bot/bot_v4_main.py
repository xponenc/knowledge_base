import asyncio

from bot_config import bot_name
from core import bot, bot_logger, init_dispatcher
from bot_v4.start_handlers import start_router
from telegram_bot.bot_v4.ai_handlers import ai_router
from telegram_bot.bot_v4.manager_handlers import manager_router
from telegram_bot.bot_v4.study_handlers import study_router
from telegram_bot.bot_v4.task_handlers import task_router


async def main():
    dp = await init_dispatcher()
    dp.include_router(start_router)
    dp.include_router(ai_router)
    dp.include_router(task_router)
    dp.include_router(study_router)
    dp.include_router(manager_router)
    bot_logger.info(f"Bot {bot_name} started")
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
