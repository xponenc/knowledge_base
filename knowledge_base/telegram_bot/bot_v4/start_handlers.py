from aiogram import Router, Bot, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.state import default_state, StatesGroup, State
from aiogram.types import BotCommand, Message, CallbackQuery, ReplyKeyboardRemove
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext

from telegram_bot.bot_config import MAIN_COMMANDS, TASK_COMMANDS, START_EMOJI, CHECK_EMOJI, QUESTION_EMOJI
from telegram_bot.bot_v4.keyboards import reply_start_keyboard
from telegram_bot.bot_v4.api_process import check_user
from telegram_bot.core import bot_logger

start_router = Router()


class AuthStates(StatesGroup):
    authorized = State()
    guest = State()


class MenuStates(StatesGroup):
    ai = State()
    task = State()
    study = State()
    manager = State()


@start_router.startup()
async def set_menu_button(bot: Bot):
    main_menu_commands = [
        BotCommand(command=f'/{key}', description=value.get("name", "")) for key, value in MAIN_COMMANDS.items()]
    system_menu_commands = [
        BotCommand(command=f'/{key}', description=value.get("name", "")) for key, value in TASK_COMMANDS.items()]
    main_menu_commands.extend(system_menu_commands)
    await bot.set_my_commands(main_menu_commands)


@start_router.message(StateFilter(None))
@start_router.message(Command('start'))
async def verify_user(message: Message, state: FSMContext):
    """Верификация пользователя при нулевом State, если пользователь неавторизован, то State остается пустым
    и остальные роутеры не сработают"""
    await state.set_state(None)
    await state.update_data(user_data={})
    print(message)
    data = await state.get_data()
    last_message = data.get("last_message")

    user_id = message.from_user.id
    bot_logger.info(f"Authorization of user with telegram ID {user_id} has started")

    profile_data = await check_user(telegram_id=user_id)

    if profile_data and profile_data["profile"] != "Anonymous":
        profile = profile_data["profile"]
        bot_logger.info(f"Access Granted for user with telegram ID {user_id}: {profile}")
        await state.update_data(user_data=profile)
        await message.answer(f"Привет, {profile['user_name']}!")
        await state.set_state(AuthStates.authorized)
    else:
        bot_logger.info(f"Access Granted for user with telegram ID {user_id}: Anonymous")
        msg = f"Привет, {message.chat.first_name or message.chat.username or message.chat.id or 'Anonymous'}"
        await message.answer(msg)
        await state.set_state(AuthStates.guest)

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{START_EMOJI}\tСтартовое меню"
        try:
            await message.bot.edit_message_text(text=text, chat_id=message.chat.id,
                                                message_id=message_id, reply_markup=None,
                                                parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    msg = ("Стартовое меню\n\n" + "\n\n".join(f'{command.get("name")} - {command.get("help_text")}'
                                              for command in
                                              list(TASK_COMMANDS.values()) + list(MAIN_COMMANDS.values())))
    answer_keyboard = await reply_start_keyboard(
        items=list(value for value in TASK_COMMANDS.values()))
    answer_message = await message.answer(msg, parse_mode=ParseMode.HTML, reply_markup=answer_keyboard)

    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": answer_keyboard.model_dump_json()
    })


@start_router.callback_query(default_state)  # default_state - это то же самое, что и StateFilter(None)
@start_router.callback_query(F.data == MAIN_COMMANDS.get("start").get("callback_data"))
async def verify_user(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await state.set_state(None)
    await state.update_data(user_data={})

    data = await state.get_data()

    last_message = data.get("last_message")

    user_id = callback.message.from_user.id
    bot_logger.info(f"Authorization of user with telegram ID {user_id} has started")

    profile_data = await check_user(telegram_id=user_id)

    if profile_data and profile_data["profile"] != "Anonymous":
        profile = profile_data["profile"]
        bot_logger.info(f"Access Granted for user with telegram ID {user_id}: {profile}")
        await state.update_data(user_data=profile)
        await callback.message.answer(f"Привет, {profile['user_name']}!")
        await state.set_state(AuthStates.authorized)
    else:
        bot_logger.info(f"Access Granted for user with telegram ID {user_id}: Anonymous")
        msg = f"Привет, {callback.message.chat.first_name or callback.message.chat.username or callback.message.chat.id or 'Anonymous'}"
        await callback.message.answer(msg)
        await state.set_state(AuthStates.guest)

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{START_EMOJI}\tСтартовое меню"
        try:
            await callback.message.bot.edit_message_text(text=text, chat_id=callback.message.chat.id,
                                                message_id=message_id, reply_markup=None,
                                                parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    msg = ("Стартовое меню\n\n" + "\n\n".join(f'{command.get("name")} - {command.get("help_text")}'
                                              for command in
                                              list(TASK_COMMANDS.values()) + list(MAIN_COMMANDS.values())))
    answer_keyboard = await reply_start_keyboard(
        items=list(value for value in TASK_COMMANDS.values()))
    answer_message = await callback.message.answer(msg, parse_mode=ParseMode.HTML, reply_markup=answer_keyboard)

    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": answer_keyboard.model_dump_json()
    })


@start_router.message(StateFilter(AuthStates.guest, AuthStates.authorized))
async def incorrect_start_options_choice(message: Message, state: FSMContext):
    bot_logger.warning(f"The user {message.from_user.id} is trying to send an invalid "
                       f"message '{message.text}' in the main menu.")

    data = await state.get_data()
    last_message = data.get("last_message")
    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{QUESTION_EMOJI}\tне понял"
        try:
            await message.bot.edit_message_text(text=text, chat_id=message.chat.id,
                                                message_id=message_id, reply_markup=None,
                                                parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            pass

    msg = "Выберите раздел:"
    answer_keyboard = await reply_start_keyboard(
        items=list(value for value in TASK_COMMANDS.values()))
    answer_message = await message.answer(text=msg,
                                          parse_mode=ParseMode.HTML,
                                          reply_markup=answer_keyboard)
    await state.update_data(last_message={
        "id": answer_message.message_id,
        "text": msg,
        "keyboard": answer_keyboard.model_dump_json()
    })
