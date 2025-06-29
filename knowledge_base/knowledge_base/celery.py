from celery import Celery
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'knowledge_base.settings')

app = Celery('knowledge_base')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()