from celery.result import AsyncResult

def get_task_status(task_id):
    """Возвращает информацию по фоновой задаче celery
    PENDING – задача ещё не выполнялась
    STARTED – уже выполняется
    SUCCESS – успешно завершена
    FAILURE – произошла ошибка
    RETRY – будет повторена
    """

    result = AsyncResult(task_id)
    return {
        "status": result.status,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
        "result": result.result if result.ready() else None,
    }