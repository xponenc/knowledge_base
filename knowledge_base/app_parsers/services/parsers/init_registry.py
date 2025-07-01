import os
import importlib
import pkgutil
import logging

from app_parsers.services.parsers.registry import register_parser, BaseWebParser

logger = logging.getLogger(__name__)


def initialize_parser_registry():
    """
    Инициализирует реестр парсеров, автоматически импортируя и регистрируя
    все классы-наследники BaseWebParser, находящиеся в папке parser_classes.

    Эта функция должна вызываться при старте как Django, так и Celery,
    поскольку AppConfig.ready() не вызывается при запуске Celery.

    Исключает повторную регистрацию при dev-запуске (если RUN_MAIN != "true").

    Логирует успешно зарегистрированные парсеры и модули.
    """
    if os.environ.get("RUN_MAIN") != "true":
        logger.debug("Пропущена регистрация парсеров: RUN_MAIN != 'true'")
        return

    parser_package = "app_parsers.services.parsers.parser_classes"
    package = importlib.import_module(parser_package)

    logger.info("🔄 Инициализация парсеров из модуля: %s", parser_package)

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{parser_package}.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
            logger.debug("Импортирован модуль парсера: %s", full_module_name)
        except Exception as e:
            logger.exception("Не удалось импортировать модуль %s: %s", full_module_name, str(e))
            continue

        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseWebParser)
                and obj is not BaseWebParser
            ):
                try:
                    register_parser(obj)
                    logger.info("✅ Зарегистрирован парсер: %s.%s", full_module_name, obj.__name__)
                except Exception as e:
                    logger.warning("⚠️ Ошибка при регистрации %s: %s", obj.__name__, str(e))
