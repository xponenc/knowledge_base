from django import forms
from django.forms import Textarea

from app_sources.source_models import NetworkDocument


class NetworkDocumentForm(forms.ModelForm):
    """Форма обновления/создания NetworkDocument"""

    class Meta:
        model = NetworkDocument
        fields = ["title", "status", "output_format", "description"]

    def __init__(self, *args, recognizers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["title"].widget.attrs.update({
                    "class": "custom-field__input custom-field__input_wide",
                    'placeholder': ' ',
                })
        self.fields["status"].widget.attrs.update({
            "class": "custom-field__input",
            'placeholder': '',
        })
        self.fields["output_format"].widget.attrs.update({
            "class": "custom-field__input",
            'placeholder': '',
        })
        self.fields['description'].widget = Textarea(attrs={
            'class': 'custom-field__input custom-field__input_wide custom-field__input_textarea'
                     ' custom-field__input_textarea-extra-toll',
            'placeholder': ' ',
        })


class NetworkDocumentStatusUpdateForm(forms.ModelForm):
    """Форма обновления статуса NetworkDocument"""

    class Meta:
        model = NetworkDocument
        fields = ['status', "output_format"]
