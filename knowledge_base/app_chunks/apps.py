import os

from django.apps import AppConfig
from knowledge_base.settings import PRODUCTION


class AppChunksConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app_chunks'

    def ready(self):
        # Избежать двойной регистрации при автоперезапуске в режиме runserver
        if not PRODUCTION and os.environ.get('RUN_MAIN') != 'true':
            return

        from app_chunks.splitters.init_registry import initialize_splitter_registry
        initialize_splitter_registry()

