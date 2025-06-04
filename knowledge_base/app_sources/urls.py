from django.urls import path
from . import views
from .views import CloudStorageListView, CloudStorageCreateView, CloudStorageDetailView, CloudStorageSyncView, \
    NetworkDocumentsMassCreateView, LocalStorageListView, LocalStorageCreateView, \
    LocalStorageDetailView, CloudStorageUpdateView, CloudStorageDeleteView, CloudStorageUpdateReportDetailView, \
    NetworkDocumentDetailView, NetworkDocumentListView, NetworkDocumentUpdateView

app_name = 'sources'


urlpatterns = [
    path('cloud_storage/', CloudStorageListView.as_view(), name='cloudstorage_list'),
    path('cloud_storage/<int:pk>', CloudStorageDetailView.as_view(), name='cloudstorage_detail'),
    path('cloud_storage/create/<int:kb_pk>', CloudStorageCreateView.as_view(), name='cloudstorage_create'),
    path('cloud_storage/<int:pk>/create/', CloudStorageUpdateView.as_view(), name='cloudstorage_update'),
    path('cloud_storage/<int:pk>/delete/', CloudStorageDeleteView.as_view(), name='cloudstorage_delete'),
    path('cloud_storage/<int:pk>/full-scan', CloudStorageSyncView.as_view(), name='cloudstorage_sync'),

    path('local_storage/', LocalStorageListView.as_view(), name='localstorage_list'),
    path('local_storage/<int:pk>', LocalStorageDetailView.as_view(), name='localstorage_detail'),
    path('local_storage/create/', LocalStorageCreateView.as_view(), name='localstorage_create'),

    path('storage-update-report/<int:pk>/', CloudStorageUpdateReportDetailView.as_view(),
         name='cloudstorageupdatereport_detail'),

    path('storage-update-report/<int:pk>/create-new-docs', NetworkDocumentsMassCreateView.as_view(),
         name='documents-mass-create'),

    path('network-document/<int:pk>', NetworkDocumentDetailView.as_view(), name='networkdocument_detail'),
    path('network-document/', NetworkDocumentListView.as_view(), name='networkdocument_list'),
    path('network-document/<int:pk>/update', NetworkDocumentUpdateView.as_view(), name='networkdocument_update'),
]
