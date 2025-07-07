import ast
import json
import re
from typing import get_origin, get_args

from django import forms
from django.core.exceptions import ValidationError


class ModelScoreTestForm(forms.Form):
    """Форма выбора url для тестирования ответов модели по контенту"""
    urls = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,  # или SelectMultiple
        choices=[],  # Заполним динамически в __init__
        required=True,
        label="Выберите до 5 тестируемых ссылок"
    )

    def __init__(self, *args, all_urls=None, **kwargs):
        super().__init__(*args, **kwargs)
        if all_urls:
            self.fields['urls'].choices = [(url, url) for url in all_urls]

    def clean_urls(self):
        selected = self.cleaned_data.get('urls')
        if len(selected) > 5:
            raise forms.ValidationError("Можно выбрать не более 5 ссылок.")
        return selected


class SplitterSelectForm(forms.Form):
    """Форма выбора сплиттера для разбиения на чанки"""
    splitters = forms.ChoiceField(label="Класс сплиттера", choices=[],
                                  widget=forms.Select(
                                      attrs={
                                          "class": "custom-field__input custom-field__input_wide",
                                          "placeholder": "",
                                      }))

    def __init__(self, *args, splitters=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not splitters:
            self.fields["splitters"].choices = [("", "--- сплиттеры не найдены ---")]
            self.fields["splitters"].disabled = True
            return

        choices = [("", "--- выберите сплиттер ---")] + [
            # (f"{cls.__module__}.{cls.__name__}", f"{cls.__name__} ({cls.name})")/
            (f"{cls.__module__}.{cls.__name__}", f"{cls.name} ({cls.__name__})")
            for cls in (splitters or [])
        ]
        self.fields["splitters"].choices = choices

    def clean_splitters(self):
        value = self.cleaned_data.get("splitters")
        if not value:
            raise ValidationError("Пожалуйста, выберите сплиттер.")
        try:
            module_name, class_name = value.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls
        except (ImportError, AttributeError, ValueError) as e:
            raise ValidationError(f"Ошибка при импорте класса: {e}")


class SplitterDynamicConfigForm(forms.Form):
    """Динамическая форма конфигурации сплиттера на основе schema"""

    def __init__(self, *args, schema: dict[str, dict] = None, initial_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.schema = schema or {}
        self.initial_config = initial_config or {}

        for field_name, meta in self.schema.items():
            field_type = meta.get("type", str)
            label = meta.get("label", field_name.replace("_", " ").capitalize())
            help_text = meta.get("help_text", "")
            required = meta.get("required", False)
            initial_value = self.initial_config.get(field_name)

            origin_type = get_origin(field_type)
            args_type = get_args(field_type)

            if field_type == bool:
                self.fields[field_name] = forms.BooleanField(
                    required=False,
                    initial=bool(initial_value),
                    label=label,
                    help_text=help_text,
                    widget=forms.CheckboxInput(attrs={"class": "custom-field__input custom-field__input_wide"})
                )

            elif field_type == int:
                self.fields[field_name] = forms.IntegerField(
                    required=required,
                    initial=initial_value,
                    label=label,
                    help_text=help_text,
                    widget=forms.NumberInput(attrs={"class": "custom-field__input custom-field__input_wide"})
                )

            elif origin_type == list:

                initial_list = initial_value or []

                if isinstance(initial_list, str):

                    try:

                        initial_list = ast.literal_eval(initial_list)

                        if not isinstance(initial_list, list):
                            initial_list = []

                    except Exception:

                        initial_list = re.split(r"[,\n;]+", initial_list)

                if not isinstance(initial_list, list):
                    initial_list = []

                # Используем тип элементов, если указан

                if args_type and args_type[0] == int:

                    initial_list = [str(int(v)) for v in initial_list if str(v).strip().isdigit()]

                else:

                    initial_list = [str(v).strip() for v in initial_list]

                initial = "\n".join(initial_list)

                self.fields[field_name] = forms.CharField(

                    required=False,

                    initial=initial,

                    label=label,

                    help_text=help_text,

                    widget=forms.Textarea(attrs={

                        "class": "custom-field__input custom-field__input_textarea custom-field__input_wide"

                    })

                )

            else:  # str и прочее
                self.fields[field_name] = forms.CharField(
                    required=required,
                    initial=initial_value or "",
                    label=label,
                    help_text=help_text,
                    widget=forms.TextInput(attrs={"class": "custom-field__input custom-field__input_wide"})
                )

    def clean(self):
        cleaned_data = super().clean()
        result = {}

        for field_name, meta in self.schema.items():
            raw_value = cleaned_data.get(field_name)
            field_type = meta.get("type", str)
            origin_type = get_origin(field_type)
            args_type = get_args(field_type)

            try:
                if field_type == bool:
                    result[field_name] = bool(raw_value)

                elif field_type == int:
                    result[field_name] = int(raw_value)

                elif origin_type == list and args_type:
                    elem_type = args_type[0]
                    parts = re.split(r"[,\n;]+", raw_value or "")
                    if elem_type == str:
                        result[field_name] = [p.strip() for p in parts if p.strip()]
                    elif elem_type == int:
                        result[field_name] = [int(p.strip()) for p in parts if p.strip()]
                    else:
                        raise ValueError(f"Неподдерживаемый тип элементов: {elem_type}")

                elif field_type == str:
                    result[field_name] = raw_value.strip() if raw_value else ""

                else:
                    raise ValueError("Неподдерживаемый тип поля")

            except Exception:
                self.add_error(field_name, f"Ошибка обработки значения: ожидается {field_type}")

        if self.errors:
            raise forms.ValidationError("Ошибки при валидации конфигурации")

        return result