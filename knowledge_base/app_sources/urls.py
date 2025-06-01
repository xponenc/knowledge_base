from django.urls import path
from . import views
from .views import CloudStorageListView, CloudStorageCreateView, CloudStorageDetailView, CloudStorageSyncView

app_name = 'sources'

urlpatterns = [
    path('cloud_storage/', CloudStorageListView.as_view(), name='cloud_storage_list'),
    path('cloud_storage/<int:pk>', CloudStorageDetailView.as_view(), name='cloud_storage_detail'),
    path('cloud_storage/create/', CloudStorageCreateView.as_view(), name='cloud_storage_create'),
    path('cloud_storage/<int:pk>/full-scan', CloudStorageSyncView.as_view(), name='cloud_storage_sync'),
]