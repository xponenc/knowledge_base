import os

import openai

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "knowledge_base.settings")
django.setup()

from django.conf import settings
from tqdm import tqdm

from app_sources.content_models import CleanedContent
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


PROMPT_TEMPLATE = """
Прочитай текст ниже и сделай по нему краткое саммари (резюме) не длиннее 9900 символов. 
Текст может содержать ошибки, постарайся понять его смысл и передать его в саммари. 
Ничего не добавляй от себя, не пиши свои выводы и мнение, только резюме по тексту. Не пиши "Вот саммари:" или "Summary:". 
Просто выведи саммари, не длиннее 4000 символов, на русском языке.


Текст:
{text}
"""

def generate_summary(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты помощник, обобщающий и корректирующий текст."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# def process_cleaned_files():
#     cleaned = CleanedContent.objects.filter(network_document__storage_id=1).select_related("network_document")
#
#     for item in tqdm(cleaned, desc="Processing summaries"):
#         # # Скипаем, если описание уже есть
#         # if item.network_document and item.network_document.description:
#         #     continue
#         # if item.local_document and item.local_document.description:
#         #     continue
#         print(item.pk, item.network_document.title)
#         try:
#             with open(item.file.path, mode="r", encoding="utf-8") as f:
#                 content_sample = f.read()[:10000]
#         except Exception as e:
#             print(f"Ошибка при чтении файла {item.id}: {e}")
#             continue
#
#         if not content_sample.strip():
#             print(f"Файл пустой: {item.id}")
#             continue
#
#         try:
#             summary = generate_summary(content_sample)
#             print(summary)
#         except Exception as e:
#             print(f"Ошибка от OpenAI: {e}")
#             continue
#
#         if item.network_document:
#             item.network_document.description = summary
#             item.network_document.save()
#         elif item.local_document:
#             item.local_document.description = summary
#             item.local_document.save()
#
#         print(f"Готово: {item.id}")
#
#
# if __name__ == "__main__":
#     process_cleaned_files()
