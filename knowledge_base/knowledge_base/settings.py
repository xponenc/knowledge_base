import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-8@pn+um^b$&-%n@ys6a*30k$jkaxeqok-a7=-)rzqu)+%2#pff'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'debug_toolbar',
    'rest_framework',
    # 'app_parsers',
    'app_core',
    'app_sources',
    'app_parsers',
    'app_chunks',
    'app_tasks',
    'app_embeddings',
    'app_chat',
    # 'app_retriever',
    "django.contrib.humanize",
    # "ckeditor",
    # "app_editor",
    'django.contrib.postgres',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'debug_toolbar.middleware.DebugToolbarMiddleware',
]

ROOT_URLCONF = 'knowledge_base.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media',
            ],
        },
    },
]

WSGI_APPLICATION = 'knowledge_base.wsgi.application'

# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases

DATABASES = {
    # 'default': {
    #     'ENGINE': 'django.db.backends.sqlite3',
    #     'NAME': BASE_DIR / 'db.sqlite3',
    # }
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv("db_name"),
        'USER': os.getenv("db_user"),
        'PASSWORD': os.getenv("db_password"),
        'HOST': 'localhost',
        'PORT': '5432',
    }

}

# kb_db_user eruy&6dw7sdajhg182736a&A&^%
# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/


STATIC_URL = '/static/'

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

TEMP_DIR = os.path.join(BASE_DIR, 'temp')  # директория для временных файлов

# CELERY_BROKER_URL = 'redis://localhost:6379/0'
# CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# REDIS+CELERY
REDIS_HOST = '127.0.0.1'
REDIS_PORT = '6379'
CELERY_BROKER_URL = 'redis://' + REDIS_HOST + ':' + REDIS_PORT + '/0'
# CELERY_BROKER_TRANSPORT_OPTION = {'visibility_timeout': 3600}
CELERY_RESULT_BACKEND = 'redis://' + REDIS_HOST + ':' + REDIS_PORT + '/0'
CELERY_TASK_SERIALIZER = 'pickle'
CELERY_RESULT_SERIALIZER = 'pickle'
CELERY_ACCEPT_CONTENT = ['pickle', 'json']
# CELERY_ACCEPT_CONTENT = ['application_json']
# CELERY_TASK_SERIALIZER = 'json'
# CELERY_RESULT_SERIALIZER = 'json'
# Celery Configuration Options
CELERY_TIMEZONE = "Europe/Samara"
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60
#
# LOGGING = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'formatters': {
#         'verbose': {
#             'format': '{asctime} [{levelname}] {name}: {message}',
#             'style': '{',
#         },
#     },
#     'handlers': {
#         'file': {
#             'level': 'INFO',
#             'class': 'logging.FileHandler',
#             'filename': os.path.join(BASE_DIR, 'logs/general.log'),
#             'formatter': 'verbose',
#         },
#
#         'console': {
#             'level': 'DEBUG',
#             'class': 'logging.StreamHandler',
#             'formatter': 'verbose',
#         },
#     },
#     'loggers': {
#         'app_sources.views': {
#             'handlers': ['file', 'console'],
#             'level': 'INFO',
#             'propagate': False,
#         },
#         # 'storages_external.webdav_storage.webdav_client': {
#         #     'handlers': ['storage_file', 'console'],
#         #     'level': 'INFO',
#         #     'propagate': False,
#         # },
#     },
# }

# Определяем, продакшен или разработка
PRODUCTION = os.getenv('DJANGO_ENV') == 'production'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'prod': {
            'format': '{asctime} [{name}:{module}:{lineno}] [{levelname}] {message}',
            'style': '{',
        },
        'dev': {
            # 'format': '{levelname} {asctime} {module} {name} line:{lineno} {process:d} {thread:d} - {message}',
            'format': '{asctime} [{name}:{module}:{lineno}] [{levelname}] {message}',
            'style': '{',
        },
    },
    'handlers': {
        'general_file': {
            'level': 'INFO',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'general.log'),
            'formatter': 'prod',
            'when': 'midnight',
            'interval': 1,
            'backupCount': 14,
            'encoding': 'utf-8',
            'delay': True,
        },
        'celery_file': {
            'level': 'INFO',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'celery.log'),
            'formatter': 'prod',
            'when': 'midnight',
            'interval': 1,
            'backupCount': 14,
            'encoding': 'utf-8',
            'delay': True,
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'dev',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['general_file'] if PRODUCTION else ['console', 'general_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'celery': {
            'handlers': ['celery_file'] if PRODUCTION else ['console', 'celery_file'],
            'level': 'INFO' if PRODUCTION else 'DEBUG',
            'propagate': False,
        },
        'urllib3': {
            'handlers': ['general_file'] if PRODUCTION else ['console', 'general_file'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['general_file'] if PRODUCTION else ['console', 'general_file'],
        'level': 'INFO',
    },
}



try:
    from .local_settings import *
except ImportError:
    try:
        from .product_settings import *
    except ImportError:
        pass

# if not DEBUG:
#     INSTALLED_APPS = [
#         *INSTALLED_APPS,
#         "debug_toolbar",
#     ]
#     MIDDLEWARE = [
#         "debug_toolbar.middleware.DebugToolbarMiddleware",
#         *MIDDLEWARE,
#     ]

