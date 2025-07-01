import logging
import os

from django.apps import AppConfig


class AppParsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app_parsers'

    def ready(self):
        if os.environ.get("RUN_MAIN") != "true":
            return
        from app_parsers.services.parsers.init_registry import initialize_parser_registry
        initialize_parser_registry()
