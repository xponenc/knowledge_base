from django import forms

from app_sources.storage_models import Storage


class StorageScanParamForm(forms.Form):
    """Форма настройки сканирования Хранилища"""
    recognize_content= forms.BooleanField(
        label="Распознавать контент",
        help_text="Автоматически выполнять распознавание файлов",
        required=True,
        initial=True
    )

    do_summarization = forms.BooleanField(
        label="Выполнить саммаризацию",
        required=False,
        initial=True,
        help_text=(
            "При включенном распознавании контента будет выполнена его саммаризация и результат сохранен в описание"
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields["recognize_content"].widget.attrs.update({
            "class": "switch",
        })
        self.fields["do_summarization"].widget.attrs.update({
            "class": "switch",
        })


class StorageTagsForm(forms.ModelForm):
    """Форма управления тегами Хранилищ(Storage)"""
    tags = forms.MultipleChoiceField(
        choices=[],
        required=False,
        widget=forms.SelectMultiple(
            attrs={
                "class": "custom-field__input custom-field__input_wide",
                "placeholder": "",
                "multiple": "multiple",
            }),
        label="Теги",
    )

    class Meta:
        model = Storage
        fields = ("tags", )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.instance and isinstance(self.instance.tags, list):
            choices = [
                (choice, choice) for choice in self.instance.tags
            ]
            if not choices:
                choices = [("", "Нет доступных тегов")]
            self.fields["tags"].choices = choices

    def clean_tags(self):
        # Список -> JSON (в Django JSONField можно сохранять как list)
        return self.cleaned_data.get("tags", [])


class StorageScanTagsForm(forms.Form):
    """Форма сканирования тегов из источников(Source) находящихся в Хранилище(Storage)"""
    scanning_depth = forms.IntegerField(
        label="Глубина сканирования тегов в тегах источников",
        min_value=1,
        error_messages={
            "min_value": "Глубина должна быть больше 1."
        }
    )


    def __init__(self, *args, recognizers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["scanning_depth"].widget.attrs.update({
                    "class": "custom-field__input",
                })
