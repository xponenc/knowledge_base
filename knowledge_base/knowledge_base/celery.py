from celery import Celery, signals
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'knowledge_base.settings')

app = Celery('knowledge_base')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@signals.worker_process_init.connect
def init_worker(**kwargs):
    from app_parsers.services.parsers.init_registry import initialize_parser_registry
    initialize_parser_registry()
