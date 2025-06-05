
# Content Recognizer Framework

Модуль `content_recognizer` предоставляет расширяемую архитектуру для распознавания текста из различных типов файлов (PDF, DOCX, изображений, Excel и т.д.) с возможностью динамического добавления новых обработчиков.

---

## 🔧 Архитектура

- **Базовый класс**:  
  Все распознаватели наследуются от абстрактного класса `ContentRecognizer`, который требует реализации метода `recognize(self)`.

- **Регистрация обработчиков**:  
  Все классы автоматически регистрируются в `recognizer_registry`, где их можно извлекать по расширению файла (`.pdf`, `.docx` и т.д.).

- **Выбор обработчика**:  
  Диспетчер `ContentRecognizerDispatcher` позволяет определить класс для заданного расширения и вызвать распознавание через `recognize_file(file_path)`.

---

## ✅ Поддерживаемые типы

| Расширение | Класс                   | Метод                |
|------------|-------------------------|----------------------|
| `.pdf`     | `PDFRecognizer`         | `pdfplumber`, OCR    |
| `.docx`    | `DOCXRecognizer`        | `python-docx`        |
| `.jpg`     | `ImageRecognizer`       | `pytesseract`        |
| `.xlsx`    | `ExcelRecognizer`       | `pandas + openpyxl`  |
| `.txt`     | `TXTRecognizer`         | `utf-8 read`         |

---

## 🧩 Как добавить новый распознаватель

1. **Создайте файл**:  
   Добавьте новый модуль в папку `content_recognizer/recognizers`, например:  
   `content_recognizer/recognizers/html_recognizer.py`

2. **Наследуйтесь от `ContentRecognizer`**:

```python
from content_recognizer.base import ContentRecognizer

class HTMLRecognizer(ContentRecognizer):
    supported_extensions = [".html", ".htm"]

    def recognize(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return {
            "text": text,
            "method": "html",
            "quality_report": {}
        }
```

3. **Убедитесь, что расширение указано в `supported_extensions`**.  
   Оно будет использоваться при маршрутизации в `recognizer_registry`.

4. **Зарегистрируйте класс вручную** в `content_recognizer/__init__.py`:

```python
from .recognizers.html_recognizer import HTMLRecognizer
from .registry import register_recognizer

register_recognizer(HTMLRecognizer)
```

---

## 🚀 Использование

```python
from content_recognizer.dispatcher import ContentRecognizerDispatcher

dispatcher = ContentRecognizerDispatcher()
result = dispatcher.recognize_file("/path/to/document.docx")
print(result["text"])
```

---

## 💡 FAQ

### Что если два обработчика поддерживают одно расширение?

При конфликте расширений будет выброшено исключение `ValueError`. Каждый обработчик должен обслуживать уникальные расширения.

### А если я хочу вручную выбрать обработчик?

```python
candidates = dispatcher.get_recognizers_for_extension(".pdf")
result = dispatcher.recognize_with(candidates[0], "/path/to/file.pdf")
```

---

## 🗂 Структура

```
content_recognizer/
├── __init__.py              # Регистрация обработчиков
├── base.py                  # Абстрактный базовый класс
├── registry.py              # Хранилище обработчиков
├── dispatcher.py            # Вызов логики по файлу
└── recognizers/
    ├── pdf_recognizer.py
    ├── docx_recognizer.py
    ├── image_recognizer.py
    ├── excel_recognizer.py
    └── txt_recognizer.py
```
