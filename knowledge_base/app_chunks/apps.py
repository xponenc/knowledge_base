import os

from django.apps import AppConfig

from app_chunks.splitters.bo_universal_splitter import BoHybridMarkdownSplitter
from app_chunks.splitters.registry import register_splitter, CHUNK_SPLITTER_REGISTRY
from app_chunks.splitters.simple_recursive_splitter import SimpleRecursiveSplitter
from knowledge_base.settings import PRODUCTION


class AppChunksConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app_chunks'

    def ready(self):
        # Избежать двойной регистрации при автоперезапуске в режиме runserver
        if not PRODUCTION and os.environ.get('RUN_MAIN') != 'true':
            return

        register_splitter(BoHybridMarkdownSplitter)
        register_splitter(SimpleRecursiveSplitter)
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Registered splitters: {list(CHUNK_SPLITTER_REGISTRY.keys())}")

