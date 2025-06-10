from django.urls import path

from app_parsers.views import ParserConfigView

app_name = "parsers"

urlpatterns = [
    path("parser/config", ParserConfigView.as_view(), name="parser_config"),
    # path("parser/<int:website_pk>/set-config", ParserSetTestConfigAsMainView.as_view(), name="set_test_parser_config_as_main")
]