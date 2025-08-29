import os
import sys
from fastapi import FastAPI, Request

from fast_api.handlers.ensemble_retriever_handler import ensemble_retriever_router
from fast_api.handlers.multi_retriever_handler import multi_retriever_router
from utils.setup_logger import setup_logger

logger = setup_logger(name=__file__, log_dir="logs/fast_api", log_file="fast_api.log")

# Добавляем корень проекта в sys.path
sys.path.append(os.path.dirname(__file__))

# Инициализация Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "knowledge_base.settings")
import django

django.setup()

from fast_api.handlers.message_handler import message_router
from fast_api.handlers.user_handler import user_router
from fast_api.handlers.multi_chain_handler import multi_chain_router
from fast_api.handlers.ensemble_chain_handler import ensemble_chain_router

# Инициализация FastAPI с метаданными
app = FastAPI(
    title="Knowledge Base AI API",
    description="API для взаимодействия с Telegram-ботом. Поддерживает обработку сообщений пользователей, генерацию ответов AI и управление профилями пользователей. Все эндпоинты доступны под префиксом `/api`.",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
)


# Middleware для логгирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Логирует входящие HTTP-запросы и ответы.

    Args:
        request (Request): Входящий HTTP-запрос.
        call_next: Функция обработки запроса.

    Returns:
        Response: Ответ сервера.
    """
    logger.info(f"Request: {request.method} {request.url} Body: {await request.body()}")
    response = await call_next(request)
    logger.info(f"Response: Status {response.status_code}")
    return response


# Подключение маршрутов FastAPI с префиксом /api
app.include_router(user_router, tags=["Пользователи"], prefix="/api/user")
app.include_router(message_router, tags=["Сообщения"], prefix="/api/message")
app.include_router(multi_chain_router, tags=["MultiChain"], prefix="/api/multi-chain")
app.include_router(ensemble_chain_router, tags=["EnsembleChain"], prefix="/api/ensemble-chain")
app.include_router(ensemble_retriever_router, tags=["EnsembleRetriever"], prefix="/api/ensemble-retriever")
app.include_router(multi_retriever_router, tags=["MultiRetriever"], prefix="/api/multi-retriever")


@app.get("/", tags=["Общее"])
async def root():
    """
    Корневой эндпоинт для проверки работы API.

    Returns:
        dict: Сообщение о статусе приложения.
    """
    return {"message": "Knowledge Base AI API is running. Документация доступна по /docs или /redoc."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fast_api.main:app", host="0.0.0.0", port=8001, reload=True)
