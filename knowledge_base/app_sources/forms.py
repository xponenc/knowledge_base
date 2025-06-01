from importlib import import_module

from django import forms
from django.core.signing import Signer

from app_sources.models import CloudStorage


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
                print(is_create)
                storage = storage_class(credentials, check_connection=is_create)
            except ValueError as e:
                raise forms.ValidationError(f"Ошибка в credentials: {e}")
        cleaned_data['credentials'] = credentials
        print(cleaned_data)
        return cleaned_data
