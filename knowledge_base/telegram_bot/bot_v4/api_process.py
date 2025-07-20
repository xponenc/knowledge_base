import aiohttp

from telegram_bot.bot_config import KB_AI_API_URL, KB_AI_API_KEY


async def check_user(telegram_id: int):
    async with aiohttp.ClientSession(headers={"Authorization": f"Bearer {KB_AI_API_KEY}"}) as session:
        async with session.get(f"{KB_AI_API_URL}/user/{telegram_id}") as resp:
            if resp.status == 200:
                return await resp.json()
            return None


async def process_message(telegram_id: int, text: str, is_user: bool, incoming_message_id: int):
    async with aiohttp.ClientSession(headers={"Authorization": f"Bearer {KB_AI_API_KEY}"}) as session:
        async with session.post(
            f"{KB_AI_API_URL}/message",
            json={
                "telegram_id": telegram_id,
                "text": text,
                "is_user": is_user,
                "incoming_message_id": incoming_message_id
            }
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            raise Exception(f"Failed to process message: {resp.status}")