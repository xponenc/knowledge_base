import os

from django.apps import AppConfig


class AppParsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app_parsers'

    def ready(self):
        # Избежать двойной регистрации при автоперезапуске
        if os.environ.get('RUN_MAIN') != 'true':
            return

        from app_parsers.services.parsers.parser_classes.bo_parser import BOWebParser
        from app_parsers.services.parsers.registry import register_parser, WEB_PARSER_REGISTRY

        register_parser(BOWebParser)

        print("Зарегистрированные парсеры:", list(WEB_PARSER_REGISTRY.keys()))