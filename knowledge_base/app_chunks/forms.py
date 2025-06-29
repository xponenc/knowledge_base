import ast
import json
import re
from typing import get_origin

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
            (f"{cls.__module__}.{cls.__name__}", f"{cls.__name__} ({cls.name})")
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
    """Динамическая форма конфигурации сплиттера"""

    def __init__(self, *args, schema: dict[str, dict] = None, initial_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.schema = schema or {}
        self.initial_config = initial_config or {}

        for field_name, meta in self.schema.items():
            field_type = meta.get("type", list[str])
            label = meta.get("label", field_name.replace("_", " ").capitalize())
            help_text = meta.get("help_text", "Вводите значения через запятую")
            if field_type == bool:
                # Обрабатываем логическое значение
                initial = bool(initial_value) if initial_value not in [None, ""] else False
                self.fields[field_name] = forms.BooleanField(
                    required=False,
                    initial=initial,
                    label=label,
                    help_text=help_text,
                    widget=forms.CheckboxInput(attrs={
                        "class": "custom-field__input custom-field__input_wide",
                        "placeholder": "",
                    })
                )
            else:
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
                    widget=forms.Textarea(attrs={
                        "class": "custom-field__input custom-field__input_textarea custom-field__input_wide",
                        "placeholder": "",
                    }),
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
                result[field_name] = bool(raw_value)

            elif field_type == str:
                result[field_name] = raw_value.strip()

            else:
                self.add_error(field_name, "Неподдерживаемый тип поля")

        if self.errors:
            raise forms.ValidationError("Ошибки при валидации конфигурации")

        return result
