
from aiogram import Dispatcher, Bot
from aiogram.fsm.storage.redis import RedisStorage
from redis.asyncio import Redis

from bot_config import bot_credentials, bot_name

from utils.setup_logger import setup_logger

# Инициализация объектов, не зависящих от asyncio
bot = Bot(token=bot_credentials)
bot_logger = setup_logger(name=__file__, log_dir="logs/telegram_bot", log_file="bot.log")

# Переменная для Dispatcher, будет инициализирована позже
dp = None


async def init_dispatcher():
    """
    Инициализация Dispatcher в асинхронном контексте.
    Вызывается, когда событийный цикл доступен.
    """
    global dp
    if dp is None:
        bot_logger.info("Initializing Dispatcher with RedisStorage")
        redis = Redis(host="localhost", port=6379, db=0)
        storage = RedisStorage(redis)
        dp = Dispatcher(storage=storage)
    return dp


async def shutdown_dispatcher():
    """
    Корректное завершение Dispatcher и закрытие соединений.
    Вызывается при завершении приложения.
    """
    global dp
    if dp is not None:
        bot_logger.info("Shutting down Dispatcher")
        await dp.storage.close()
        dp = None


def get_dispatcher():
    """
    Синхронная функция для получения Dispatcher.
    Если dp не инициализирован, поднимает исключение, чтобы напомнить о необходимости вызова init_dispatcher.
    """
    if dp is None:
        raise RuntimeError("Dispatcher not initialized. Call init_dispatcher() in an async context first.")
    return dp
