import ast
import json
import re
from typing import Optional, List, Type, get_origin

from django import forms

from app_parsers.services.parsers.base import BaseWebParser
from app_parsers.services.parsers.dispatcher import WebParserDispatcher


class ParserDynamicConfigForm(forms.Form):
    """Динамическая форма конфигурации парсера"""

    def __init__(self, *args, schema: dict[str, dict] = None, initial_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.schema = schema or {}
        self.initial_config = initial_config or {}

        for field_name, meta in self.schema.items():
            field_type = meta.get("type", list[str])
            label = meta.get("label", field_name.replace("_", " ").capitalize())
            help_text = meta.get("help_text", "Вводите значения через запятую")

            # Проверяем, является ли field_type параметризованным типом list
            is_list_type = get_origin(field_type) is list
            if is_list_type:
                initial_value = self.initial_config.get(field_name, [])
                if isinstance(initial_value, str):
                    try:
                        # Пытаемся десериализовать строку как Python список
                        initial_value = ast.literal_eval(initial_value) if initial_value else []
                        if not isinstance(initial_value, list):
                            initial_value = []
                    except (ValueError, SyntaxError):
                        initial_value = initial_value.splitlines() if initial_value else []
                initial = ", ".join(str(v) for v in initial_value)
            else:
                initial = str(self.initial_config.get(field_name, ""))

            self.fields[field_name] = forms.CharField(
                required=False,
                initial=initial,
                widget=forms.Textarea(attrs={"rows": 8}),
                label=label,
                help_text=help_text
            )

    def clean(self):
        cleaned_data = super().clean()
        result = {}

        for field_name, meta in self.schema.items():
            raw_value = cleaned_data.get(field_name)
            field_type = meta.get("type", list[str])

            if field_type == list[str]:
                try:
                    # Попробуем сначала распарсить как JSON
                    try:
                        value = json.loads(raw_value)
                        if not isinstance(value, list):
                            raise ValueError("JSON должен быть списком")
                    except json.JSONDecodeError:
                        # Если это не JSON — разбиваем по , ; и \n
                        parts = re.split(r'[,\n;]+', raw_value)
                        value = [part.strip() for part in parts if part.strip()]
                    result[field_name] = value
                except Exception as e:
                    self.add_error(field_name, "Некорректный список строк")

            elif field_type == list[int]:
                try:
                    value = [int(line.strip()) for line in raw_value.splitlines() if line.strip()]
                    result[field_name] = value
                except ValueError:
                    self.add_error(field_name, "Должны быть целые числа, по одному на строку")

            elif field_type == bool:
                if raw_value.lower() in ("true", "1", "yes", "да"):
                    result[field_name] = True
                elif raw_value.lower() in ("false", "0", "no", "нет"):
                    result[field_name] = False
                else:
                    self.add_error(field_name, "Введите True/False или Да/Нет")

            elif field_type == str:
                result[field_name] = raw_value.strip()

            else:
                self.add_error(field_name, "Неподдерживаемый тип поля")

        if self.errors:
            raise forms.ValidationError("Ошибки при валидации конфигурации")

        return result

# class ParserDynamicConfigForm(forms.Form):
#     """Динамическая форма конфигурации парсера"""
#
#     def __init__(self, *args, schema: dict[str, dict] = None, initial_config=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.schema = schema or {}
#         self.initial_config = initial_config or {}
#
#         for field_name, meta in self.schema.items():
#             field_type = meta.get("type", list[str])
#             label = meta.get("label", field_name.replace("_", " ").capitalize())
#             help_text = meta.get("help_text", "Вводите по одному значению на строку")
#
#             # Обработка initial_config: проверяем, является ли значение списком или строкой
#             initial_value = self.initial_config.get(field_name, [])
#             if isinstance(field_type, type) and issubclass(field_type, list):
#                 if isinstance(initial_value, str):
#                     try:
#                         # Пытаемся десериализовать строку как JSON или Python список
#                         initial_value = ast.literal_eval(initial_value) if initial_value else []
#                         if not isinstance(initial_value, list):
#                             initial_value = []
#                     except (ValueError, SyntaxError):
#                         initial_value = initial_value.splitlines() if initial_value else []
#                 initial = "\n".join(str(v) for v in initial_value)
#             else:
#                 initial = str(initial_value) if initial_value else ""
#
#             self.fields[field_name] = forms.CharField(
#                 required=False,
#                 initial=initial,
#                 widget=forms.Textarea(attrs={"rows": 3}),
#                 label=label,
#                 help_text=help_text
#             )
#
#     def clean(self):
#         cleaned_data = super().clean()
#         result = {}
#
#         for field_name, meta in self.schema.items():
#             raw_value = cleaned_data.get(field_name, "")
#             field_type = meta.get("type", list[str])
#
#             if field_type == list[str]:
#                 try:
#                     # Сначала пытаемся распарсить как JSON
#                     try:
#                         value = json.loads(raw_value)
#                         if not isinstance(value, list):
#                             raise ValueError("JSON должен быть списком")
#                     except json.JSONDecodeError:
#                         # Если не JSON, пытаемся интерпретировать как Python список
#                         try:
#                             value = ast.literal_eval(raw_value)
#                             if not isinstance(value, list):
#                                 raise ValueError("Должно быть списком")
#                         except (ValueError, SyntaxError):
#                             # Если не Python список, разбиваем по разделителям
#                             parts = re.split(r'[,\n;]+', raw_value)
#                             value = [part.strip() for part in parts if part.strip()]
#                     result[field_name] = value
#                 except Exception as e:
#                     self.add_error(field_name, "Некорректный список строк")
#
#             elif field_type == list[int]:
#                 try:
#                     value = [int(line.strip()) for line in raw_value.splitlines() if line.strip()]
#                     result[field_name] = value
#                 except ValueError:
#                     self.add_error(field_name, "Должны быть целые числа, по одному на строку")
#
#             elif field_type == bool:
#                 raw_value_lower = raw_value.lower()
#                 if raw_value_lower in ("true", "1", "yes", "да"):
#                     result[field_name] = True
#                 elif raw_value_lower in ("false", "0", "no", "нет"):
#                     result[field_name] = False
#                 else:
#                     self.add_error(field_name, "Введите True/False или Да/Нет")
#
#             elif field_type == str:
#                 result[field_name] = raw_value.strip()
#
#             else:
#                 self.add_error(field_name, "Неподдерживаемый тип поля")
#
#         if self.errors:
#             raise forms.ValidationError("Ошибки при валидации конфигурации")
#
#         return result


class TestParseForm(forms.Form):
    """Форма для тестового парсинга одного url выбранным parser"""
    parser = forms.ChoiceField(choices=[], label="Выберите парсер")
    url = forms.URLField(label="URL для теста", required=True)

    def __init__(self, *args, parsers: Optional[List[Type[BaseWebParser]]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        choices = [("", "— Выберите парсер —")] + [
            (f"{cls.__module__}.{cls.__name__}", cls.__name__)
            for cls in (parsers or [])
        ]
        self.fields["parser"].choices = choices
        self.fields["parser"].widget.attrs.update({
            "class": "form-select",
        })

    def clean_parser(self):
        value = self.cleaned_data["parser"]
        dispatcher = WebParserDispatcher()
        try:
            parser = dispatcher.get_by_class_name(value)
            return parser
        except ValueError as e:
            raise forms.ValidationError(str(e))


class BulkParseForm(forms.Form):
    """Форма для массового парсинга списка URL выбранным парсером"""
    parser = forms.ChoiceField(choices=[], label="Выберите парсер")
    urls = forms.CharField(
        label="Список URL (формат: JSON или просто через , ; \\n)",
        widget=forms.Textarea(attrs={"rows": 6}),
        required=True,
    )

    def __init__(self, *args, parsers: Optional[List[Type[BaseWebParser]]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        choices = [
            (f"{cls.__module__}.{cls.__name__}", cls.__name__)
            for cls in (parsers or [])
        ]
        self.fields["parser"].choices = choices
        self.fields["parser"].widget.attrs.update({
            "class": "form-select",
        })

    def clean_parser(self):
        value = self.cleaned_data["parser"]
        try:
            module_name, class_name = value.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls
        except (ImportError, AttributeError, ValueError) as e:
            raise forms.ValidationError(f"Не удалось загрузить парсер: {e}")

    def clean_urls(self):
        raw_input = self.cleaned_data["urls"].strip()

        urls = []

        # Пытаемся распарсить как JSON-массив
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                urls = parsed
        except json.JSONDecodeError:
            # Если не JSON — разбиваем вручную по строкам и разделителям
            url_candidates = [
                url.strip()
                for part in raw_input.splitlines()
                for url in part.replace(';', '\n').replace(',', '\n').split('\n')
            ]
            urls = [url for url in url_candidates if url]

        # Валидация URL-ов
        valid_urls = [url for url in urls if url.startswith("http://") or url.startswith("https://")]
        if not valid_urls:
            raise forms.ValidationError("Не удалось найти ни одного корректного URL.")
        return valid_urls
