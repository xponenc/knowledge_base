import os

from django.apps import AppConfig

from app_chunks.splitters.bo_universal_splitter import BoHybridMarkdownSplitter
from app_chunks.splitters.registry import register_splitter, CHUNK_SPLITTER_REGISTRY


class AppChunksConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app_chunks'

    def ready(self):
        # Избежать двойной регистрации при автоперезапуске
        if os.environ.get('RUN_MAIN') != 'true':
            return

        register_splitter(BoHybridMarkdownSplitter)
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Registered splitters: {list(CHUNK_SPLITTER_REGISTRY.keys())}")
        print(f"Registered splitters: {list(CHUNK_SPLITTER_REGISTRY.keys())}")

