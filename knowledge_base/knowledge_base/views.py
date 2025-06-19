from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_GET

from app_sources.report_models import CloudStorageUpdateReport
from app_sources.storage_models import CloudStorage


def get_parent_instance(parent_type: str, parent_pk: int):

    model_map = {
        "cloudstorage": CloudStorage,
        "cloudstorageupdatereport": CloudStorageUpdateReport,
        # "source": SourceModel,
        # добавь свои модели
    }

    model_class = model_map.get(parent_type.lower())
    if not model_class:
        raise Http404("Неизвестный тип родителя задачи")

    return get_object_or_404(model_class, pk=parent_pk)

@login_required
@require_GET
def get_celery_task_progress(request, parent_type, parent_pk, task_pk, ):
    """Отображает прогресс выполнения фоновой задачи Celery"""
    task_id = str(task_pk)
    parent_obj = get_parent_instance(parent_type, parent_pk)
    try:
        running_background_tasks = parent_obj.running_background_tasks
    except AttributeError:
        raise Http404(f"Не задан running_background_tasks для объекта фоновой задачи {parent_obj}")

    task_name = running_background_tasks.get(task_id)
    if not task_name:
        raise Http404(f"Фоновая задача id {task_id} не найдена в объекте фоновой задачи")

    try:
        url = parent_obj.get_absolute_url()
    except AttributeError:
        raise Http404(f"Не задан метод get_absolute_url для объекта фоновой задачи {parent_obj}")

    # task_result = get_task_status(task_id)
    context = {
        "task_id": task_id,
        "task_name": task_name,
        "task_object": parent_obj,
        "task_object_url": url,
        "next_step_url": url,
        "task_object_name": str(parent_obj),
        # "task_status": task_result,
    }

    return render(request=request, template_name="celery_task_progress.html", context=context)