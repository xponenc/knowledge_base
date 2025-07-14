from django import forms


class SystemInstructionForm(forms.Form):
    """форма редактирования системной инструкции для ai"""

    system_instruction = forms.CharField(
        label="Системная инструкция",
        widget=forms.Textarea(attrs={
            'class': 'custom-field__input custom-field__input_textarea-extra-toll custom-field__input_wide',
            'placeholder': ' '
        })
    )
