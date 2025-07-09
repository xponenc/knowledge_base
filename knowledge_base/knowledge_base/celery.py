from celery import Celery, signals
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'knowledge_base.settings')

app = Celery('knowledge_base')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@signals.worker_process_init.connect
def init_worker(**kwargs):
    # Инициализация классов парсеров
    from app_parsers.services.parsers.init_registry import initialize_parser_registry
    initialize_parser_registry()
    # Инициализация классов сплиттеров
    from app_chunks.splitters.init_registry import initialize_splitter_registry
    initialize_splitter_registry()
