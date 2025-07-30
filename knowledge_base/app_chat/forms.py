import csv
from io import TextIOWrapper, StringIO

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


class KnowledgeBaseBulkTestForm(forms.ModelForm):
    """Форма тестирования базы знаний списком вопросов"""

    questions_text = forms.CharField(
        widget=forms.Textarea(attrs={
            "class": "custom-field__input custom-field__input_wide custom-field__input_textarea-extra-toll"
        }),
        required=False,
        label="Введите тестовые вопросы (по одному вопросу на строку)",
        help_text="Вопросы будут разделены по переводу строки"
    )

    csv_file = forms.FileField(
        required=False,
        label="Загрузите CSV файл",
        help_text="CSV файл в кодировке utf-8 с вопросами в первой колонке (разделитель ;)"
    )

    class Meta:
        model = KnowledgeBase
        fields = ("llm", "retriever_scheme")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['llm'].widget.attrs.update({
            'class': 'custom-field__input',
        })

        self.fields['retriever_scheme'].widget.attrs.update({
            'class': 'custom-field__input',
        })

    def clean(self):
        cleaned_data = super().clean()
        questions_text = cleaned_data.get("questions_text")
        csv_file = cleaned_data.get("csv_file")

        if not questions_text and not csv_file:
            raise forms.ValidationError(
                "Пожалуйста, укажите вопросы либо в текстовом поле, либо загрузив CSV-файл.")

        if questions_text and csv_file:
            raise forms.ValidationError("Укажите вопросы только в одном из полей: текстовом или CSV-файле.")

        if csv_file:
            if not csv_file.name.endswith('.csv'):
                raise forms.ValidationError("Загруженный файл должен быть в формате CSV.")
            try:
                file_content = csv_file.read().decode('utf-8')
                cleaned_data['csv_content'] = file_content
                csv_reader = csv.reader(StringIO(file_content), delimiter=";")
                for row in csv_reader:
                    if not row or not row[0].strip():
                        raise forms.ValidationError("CSV-файл должен содержать хотя бы один столбец с вопросами.")
                    break
            except Exception as e:
                raise forms.ValidationError(f"Ошибка при чтении CSV-файла: {str(e)}")

        return cleaned_data

    def get_questions(self):
        questions_text = self.cleaned_data.get("questions_text")
        csv_content = self.cleaned_data.get("csv_content")

        if questions_text:
            return [q.strip() for q in questions_text.split('\n') if q.strip()]
        elif csv_content:
            questions = []
            csv_reader = csv.reader(StringIO(csv_content), delimiter=';')
            for row in csv_reader:
                if row and row[0].strip():
                    questions.append(row[0].strip())
            return questions
        return []