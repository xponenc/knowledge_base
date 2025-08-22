from django import forms

from app_ai_assistants.models import Assistant


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
