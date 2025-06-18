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