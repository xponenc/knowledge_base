from typing import Any

from django import forms
from pydantic import ValidationError
from pydantic.v1.fields import Undefined

from app_ai_assistants.models import Assistant, Block
from app_ai_assistants.services.block_model_validation import validate_block_config, BLOCK_TYPE_TO_SCHEMA


class AssistantTypeForm(forms.ModelForm):
    """Форма выбора типа AI помощника"""

    class Meta:
        model = Assistant
        fields = ["type", ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['type'].widget.attrs.update({
            'class': 'custom-field__input custom-field__input_wide',
            'placeholder': '',
        })


class BlockConfigForm(forms.ModelForm):
    class Meta:
        model = Block
        fields = ["config"]
        widgets = {"config": forms.HiddenInput()}  # чтобы Django не пытался рендерить и валидировать это поле

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        block = self.instance
        schema = BLOCK_TYPE_TO_SCHEMA.get(block.block_type)
        if not schema:
            return

        # Подменяем стандартное поле config на динамические
        # self.fields.pop("config")

        for field_name, model_field in schema.model_fields.items():
            value = self.instance.config.get(field_name, None)
            if value is Undefined:
                value = None

            field_type = model_field.annotation
            if field_type == bool:
                form_field = forms.BooleanField(
                    required=False,
                    initial=value,
                    widget=forms.CheckboxInput(attrs={"class": "switch"})
                )
            elif field_type in (int, float):
                form_field = forms.FloatField(
                    required=not model_field.is_required,
                    initial=value,
                    widget=forms.NumberInput(attrs={"class": "custom-field__input"})
                )
            else:
                if value and len(str(value)) > 60:
                    widget = forms.Textarea(attrs={"class": "custom-field__input custom-field__input_wide custom-field__input_textarea"})
                else:
                    widget = forms.TextInput(attrs={"class": "custom-field__input"})
                form_field = forms.CharField(
                    required=not model_field.is_required,
                    initial=value,
                    widget=widget
                )

            form_field.label = model_field.title or field_name
            form_field.help_text = model_field.description or ""
            self.fields[field_name] = form_field

    def clean(self):
        cleaned_data = super().clean()
        schema = BLOCK_TYPE_TO_SCHEMA.get(self.instance.block_type)
        if not schema:
            return cleaned_data

        try:
            validated = schema(**cleaned_data)
            self.cleaned_data["config"] = validated.model_dump()
        except ValidationError as e:
            raise forms.ValidationError(e.errors())
        return self.cleaned_data

    def save(self, commit=True):
        schema = BLOCK_TYPE_TO_SCHEMA.get(self.instance.block_type)
        data = {f: self.cleaned_data[f] for f in self.fields}
        validated = schema(**data)
        self.instance.config = validated.model_dump()
        if commit:
            self.instance.save()
        return self.instance
