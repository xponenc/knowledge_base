import importlib
import pkgutil
import logging

from app_chunks.splitters.base import BaseSplitter
from app_chunks.splitters.registry import register_splitter, CHUNK_SPLITTER_REGISTRY

logger = logging.getLogger(__name__)


def initialize_splitter_registry():
    splitters_package = "app_chunks.splitters.splitter_classes"
    package = importlib.import_module(splitters_package)

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{splitters_package}.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
            logger.debug("Successfully imported splitter module: %s", full_module_name)
        except Exception as e:
            logger.exception("Could not import module %s: %s", full_module_name, str(e))
            continue

        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseSplitter)
                and obj is not BaseSplitter
            ):
                register_splitter(obj)

    logger.info(f"Registered splitters: {list(CHUNK_SPLITTER_REGISTRY)}")
