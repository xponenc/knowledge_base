from aiogram import Router, Bot, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.state import default_state, StatesGroup, State
from aiogram.types import BotCommand, Message, CallbackQuery, ReplyKeyboardRemove
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext

from telegram_bot.bot_config import MAIN_COMMANDS, TASK_COMMANDS, START_EMOJI, CHECK_EMOJI
from telegram_bot.bot_v4.keyboards import reply_start_keyboard
from telegram_bot.bot_v4.api_process import check_user
from telegram_bot.core import bot_logger

start_router = Router()


class AuthStates(StatesGroup):
    authorized = State()
    guest = State()


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
    await state.update_data(user={})

    data = await state.get_data()
    # print(data)
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
        await message.answer("Привет, Anonymous! Создаю новую сессию.")
        await state.set_state(AuthStates.authorized)

    if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
        message_id = last_message.get("id")
        text = last_message.get("text")
        text += f"\n\n{CHECK_EMOJI} {START_EMOJI} Стартовое меню"
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

#
# @start_router.callback_query(default_state)  # default_state - это то же самое, что и StateFilter(None)
# @start_router.callback_query(F.data == MAIN_COMMANDS.get("start").get("callback_data"))
# async def verify_user(callback: CallbackQuery, state: FSMContext):
#     await callback.answer()
#     # await state.clear()
#     await state.set_state(None)
#     await state.update_data(task_data={})
#
#     data = await state.get_data()
#     # print(data)
#     last_message = data.get("last_message")
#
#     user_id = callback.from_user.id
#     bot_logger.info(f"Authorization of user with telegram ID {user_id} has started")
#     verified_user = await get_user_context(telegram_id=user_id)
#     if not verified_user:
#         await callback.message.answer("Доступ запрещен", reply_markup=ReplyKeyboardRemove())
#         bot_logger.warning(f"Access Denied for user with telegram ID {user_id}")
#
#     else:
#         if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
#             message_id = last_message.get("id")
#             text = last_message.get("text")
#             text += f"\n\n{CHECK_EMOJI} {START_EMOJI} Стартовое меню"
#             try:
#                 await callback.message.bot.edit_message_text(text=text, chat_id=callback.message.chat.id,
#                                                              message_id=message_id, reply_markup=None,
#                                                              parse_mode=ParseMode.HTML)
#             except TelegramBadRequest:
#                 pass
#         await state.set_state(StartState.waiting_start_menu_selection)
#         await state.update_data(user_data=verified_user)
#         bot_logger.info(f"Access Granted for user with telegram ID {user_id}: {verified_user}")
#
#         msg = ("Стартовое меню\n\n" + "\n\n".join(f'{command.get("name")} - {command.get("help_text")}'
#                                                   for command in
#                                                   list(TASK_COMMANDS.values()) + list(MAIN_COMMANDS.values())))
#         answer_keyboard = await reply_start_keyboard(
#             items=list(value for value in TASK_COMMANDS.values()))
#         answer_message = await callback.message.answer(msg, parse_mode=ParseMode.HTML, reply_markup=answer_keyboard)
#
#         await state.update_data(last_message={
#             "id": answer_message.message_id,
#             "text": msg,
#             "keyboard": answer_keyboard.model_dump_json()
#         })

#
# # Отлов неправильных команд в главном меню
# @start_router.message(StartState.waiting_start_menu_selection,
#                       ~F.text.in_([f"/{command}" for command in list(TASK_COMMANDS) + list(MAIN_COMMANDS)]))
# async def incorrect_start_options_choice(message: Message, state: FSMContext):
#     bot_logger.warning(f"The user {message.from_user.id} is trying to send an invalid "
#                        f"command '{message.text}' in the main menu.")
#
#     data = await state.get_data()
#     # print(data)
#     last_message = data.get("last_message")
#     if last_message:  # Сброс клавиатуры последнего сообщения и отметка о выбранном варианте
#         message_id = last_message.get("id")
#         text = last_message.get("text")
#         text += f"\n\n{CHECK_EMOJI} Получена некорректная команда"
#         try:
#             await message.bot.edit_message_text(text=text, chat_id=message.chat.id,
#                                                 message_id=message_id, reply_markup=None,
#                                                 parse_mode=ParseMode.HTML)
#         except TelegramBadRequest:
#             pass
#
#     msg = "Я не знаю такую команду, выберите из:"
#     answer_keyboard = await reply_start_keyboard(
#         items=list(value for value in TASK_COMMANDS.values()))
#     answer_message = await message.answer(text=msg,
#                                           parse_mode=ParseMode.HTML,
#                                           reply_markup=answer_keyboard)
#     await state.update_data(last_message={
#         "id": answer_message.message_id,
#         "text": msg,
#         "keyboard": answer_keyboard.model_dump_json()
#     })
