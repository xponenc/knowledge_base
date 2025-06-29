from django import forms


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
        choices = [
            (f"{cls.__module__}.{cls.__name__}", f"{cls.__name__}({cls.name})")
            for cls in (splitters or [])
        ]
        self.fields["splitters"].choices = choices

    def clean_splitters(self):
        value = self.cleaned_data["splitters"]
        module_name, class_name = value.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls
