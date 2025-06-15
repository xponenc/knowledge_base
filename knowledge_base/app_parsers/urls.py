from django.urls import path

from app_parsers.views import ParserConfigView, TestParserDetailView, ParserSetTestAsMainView, MainParserDetailView

app_name = "parsers"


urlpatterns = [
    path("parser/config", ParserConfigView.as_view(), name="parser_config"),

    path("test-parser/<int:pk>", TestParserDetailView.as_view(), name="testparser_detail"),
    path("test-parser/<int:pk>/set-as-main", ParserSetTestAsMainView.as_view(), name="set_test_parser_as_main"),

    path("main-parser/<int:pk>", MainParserDetailView.as_view(), name="mainparser_detail"),
]