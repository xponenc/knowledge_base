from django import forms

from app_core.models import KnowledgeBase


class SystemChatInstructionForm(forms.ModelForm):
    """форма редактирования системной инструкции для ai"""

    class Meta:
        model = KnowledgeBase
        fields = ("system_instruction", "llm", )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['system_instruction'].widget = forms.Textarea(attrs={
            'class': 'custom-field__input custom-field__input_textarea-extra-toll custom-field__input_wide',
            'placeholder': ' '
        })
        self.fields['llm'].widget.attrs.update({
            'class': 'custom-field__input custom-field__input_wide',
            'placeholder': '',
        })
