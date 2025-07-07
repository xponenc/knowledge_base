from django import forms
from django.forms import Textarea

from app_core.models import KnowledgeBase


class KnowledgeBaseForm(forms.ModelForm):
    """Форма создания/редактирования KnowledgeBase"""
    class Meta:
        model = KnowledgeBase
        fields = ['name', 'engine', 'description', 'owners', 'logo']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['name'].widget.attrs.update({
            'class': 'custom-field__input custom-field__input_wide',
            'placeholder': ' ',
        })
        self.fields['engine'].widget.attrs.update({
            'class': 'custom-field__input custom-field__input_wide',
            'placeholder': '',
        })
        self.fields['description'].widget = Textarea(attrs={
            'class': 'custom-field__input custom-field__input_wide custom-field__input_textarea',
            'placeholder': ' ',
        })
        self.fields['owners'].widget.attrs.update({
            'class': 'custom-field__input custom-field__input_wide',
            'placeholder': '',
        })
        self.fields['logo'].widget.attrs.update({
            'class': 'custom-field__input custom-field__input_wide',
            'placeholder': ' ',
        })
