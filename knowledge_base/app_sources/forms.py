from importlib import import_module

from django import forms
from django.core.signing import Signer

from app_sources.source_models import NetworkDocument
from app_sources.storage_models import CloudStorage


class CloudStorageForm(forms.ModelForm):
    """Форма создания объекта модели Облачное хранилище"""
    url = forms.URLField()
    root_path = forms.CharField()
    auth_type = forms.ChoiceField(choices=[('token', 'Токен'), ('basic', 'Логин/Пароль')])
    token = forms.CharField(required=False)
    username = forms.CharField(required=False)
    password = forms.CharField(widget=forms.PasswordInput, required=False)

    class Meta:
        model = CloudStorage
        fields = ['name', 'api_type', 'credentials', ]
    #     widgets = {
    #         'credentials': forms.Textarea(attrs={'rows': 4}),
    #     }

    def __init__(self, *args, **kwargs):
        super(CloudStorageForm, self).__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs.update({
                "autocomplete": "off",
                "autocorrect": "off",
                "autocapitalize": "off",
                "spellcheck": "false",
            })

    def clean(self):
        cleaned_data = super().clean()
        signer = Signer()

        credentials = {
            'url': cleaned_data.get('url'),
            'root_path': cleaned_data.get('root_path'),
            'auth_type': cleaned_data.get('auth_type'),
        }
        if cleaned_data.get('auth_type') == 'token':
            if not cleaned_data.get('token'):
                raise forms.ValidationError("Для auth_type='token' требуется токен")
            credentials['token'] = signer.sign(cleaned_data.get('token'))
        else:
            if not cleaned_data.get('username') or not cleaned_data.get('password'):
                raise forms.ValidationError("Для auth_type='basic' требуются имя пользователя и пароль")
            credentials['username'] = cleaned_data.get('username')
            credentials['password'] = signer.sign(cleaned_data.get('password'))

        # credentials = self.cleaned_data.get('credentials')
        api_type = self.cleaned_data.get('api_type')
        if api_type in CloudStorage.STORAGE_CLASSES:
            try:
                module_path, class_name = CloudStorage.STORAGE_CLASSES[api_type].rsplit('.', 1)
                module = import_module(module_path)
                storage_class = getattr(module, class_name)
                is_create = not self.instance.pk
                storage = storage_class(credentials, check_connection=is_create)
            except ValueError as e:
                raise forms.ValidationError(f"Ошибка в credentials: {e}")
        cleaned_data['credentials'] = credentials
        return cleaned_data


class ContentRecognizerForm(forms.Form):
    recognizer = forms.ChoiceField(label="Метод распознавания", choices=[],
                                   widget=forms.Select(
                                       attrs={
                                           "class": "custom-field__input custom-field__input_wide",
                                           "placeholder": "",
                                       }))

    def __init__(self, *args, recognizers=None, **kwargs):
        super().__init__(*args, **kwargs)
        choices = [
            (f"{cls.__module__}.{cls.__name__}", cls.__name__)
            for cls in (recognizers or [])
        ]
        self.fields["recognizer"].choices = choices
        # self.fields["recognizer"].widget.attrs.update({
        #             "class": "custom-field__input",
        #         })
        # self.fields['description'].widget.attrs.update({
        #     'class': 'custom-field__input custom-field__input_textarea custom-field__input_wide',
        #     'placeholder': ' ',
        # })

    def clean_recognizer(self):
        value = self.cleaned_data["recognizer"]
        module_name, class_name = value.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls


class CleanedContentEditorForm(forms.Form):
    """Форма для редактирования содержимого файла CleanedContent аттрибут файл через онлайн маркдаун редактор"""
    content = forms.CharField(widget=forms.HiddenInput())  # Поле для Markdown


