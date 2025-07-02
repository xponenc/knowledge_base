import logging
import os

from django.apps import AppConfig

from knowledge_base.settings import PRODUCTION


class AppParsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app_parsers'

    def ready(self):
        # Избежать двойной регистрации при автоперезапуске в режиме runserver
        if PRODUCTION:
            return
        from app_parsers.services.parsers.init_registry import initialize_parser_registry
        initialize_parser_registry()
