from django.urls import path

from app_tasks.views import TaskForSourceDetailView

app_name = "tasks"

urlpatterns = [
    path("task_for_source_detail/<int:pk>", TaskForSourceDetailView.as_view(), name="task_for_source_detail"),
]
