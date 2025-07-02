import os
import importlib
import pkgutil
import logging

from app_parsers.services.parsers.registry import register_parser, BaseWebParser, WEB_PARSER_REGISTRY

logger = logging.getLogger(__name__)


def initialize_parser_registry():
    parser_package = "app_parsers.services.parsers.parser_classes"
    package = importlib.import_module(parser_package)

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{parser_package}.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
            logger.debug("Successfully imported parser module: %s", full_module_name)
        except Exception as e:
            logger.exception("Could not import module %s: %s", full_module_name, str(e))
            continue

        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseWebParser)
                and obj is not BaseWebParser
            ):
                register_parser(obj)

    logger.info(f"Registered parsers: {list(WEB_PARSER_REGISTRY)}")
