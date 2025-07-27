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


class KBRandomTestForm(forms.Form):
    """
    Динамическая форма: выбор хранилищ и количество тестовых вопросов.
    Формирует поля группами: WebSite, CloudStorage, LocalStorage, URLBatch.
    """

    def __init__(self, *args, kb: KnowledgeBase = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.grouped_fields = []  # список групп с полями для шаблона

        if kb is None:
            return

        for model_name, related_name, group_label in [
            ("website", "website_set", "Сайты"),
            ("cloudstorage", "cloudstorage_set", "Облачные хранилища"),
            ("localstorage", "localstorage_set", "Локальные хранилища"),
            ("urlbatch", "urlbatch_set", "Пакеты ссылок"),
        ]:
            group = []
            storages = getattr(kb, related_name).all()

            for storage in storages:
                checkbox_name = f"use_{model_name}_{storage.pk}"
                count_name = f"count_{model_name}_{storage.pk}"

                self.fields[checkbox_name] = forms.BooleanField(
                    label=storage.name, required=False,
                    widget=forms.CheckboxInput(attrs={"class": "switch"})
                )
                self.fields[count_name] = forms.IntegerField(
                    label="Кол-во вопросов", min_value=1, required=False,
                    widget=forms.NumberInput(attrs={
                        "class": "custom-field__input",
                        "placeholder": "",
                    })
                )
                group.append(checkbox_name)
                group.append(count_name)
            if group:
                self.grouped_fields.append((group_label, group))

