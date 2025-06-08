from django.urls import path

from app_parsers.views import ParserConfigView

app_name = "parsers"

urlpatterns = [
    path("parser/config", ParserConfigView.as_view(), name="parser_config")
]