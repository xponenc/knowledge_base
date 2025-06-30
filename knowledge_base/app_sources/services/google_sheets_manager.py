import os
import random
import sys
import time

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from urllib.parse import urljoin, quote
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from utils.setup_logger import setup_logger

# Добавляем корень проекта в sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

logger = setup_logger(__name__, log_dir="logs/google_sheets", log_file="sheets_operations.log")


def with_retry(func, retries=5):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            wait = (2 ** i) + random.uniform(0, 1)
            logger.warning(f"Retry {i + 1}/{retries}: {e} (wait {wait:.2f}s)")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

class GoogleSheetsManager:
    def __init__(self, credentials_file, short_sheet_name="DocScanner_Summary", full_sheet_name="DocScanner_FullSummary", parent_folder_id=None):
        self.scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets"
        ]
        if not os.path.isfile(credentials_file):
            logger.error(f"Файл учетных данных не найден: {credentials_file}")
            raise FileNotFoundError(f"Credentials file not found: {credentials_file}")

        logger.info(f"Используется файл учетных данных: {credentials_file}")
        self.credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, self.scope)
        self.client = gspread.authorize(self.credentials)
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        self.sheets_service = build('sheets', 'v4', credentials=self.credentials)

        self.short_sheet, self.short_spreadsheet_id, self.short_shared_link = self._get_or_create_sheet(
            short_sheet_name, parent_folder_id)
        self.full_sheet, self.full_spreadsheet_id, _ = self._get_or_create_sheet(
            full_sheet_name, parent_folder_id)

        self.short_sheet_name = short_sheet_name
        self.full_sheet_name = full_sheet_name

    def _get_or_create_sheet(self, sheet_name, folder_id=None):
        try:
            logger.info(f"Попытка открыть таблицу: {sheet_name}")
            sheet = self.client.open(sheet_name).sheet1
            spreadsheet_id = sheet.spreadsheet.id
            file = self.drive_service.files().get(fileId=spreadsheet_id, fields='webViewLink').execute()
            return sheet, spreadsheet_id, file.get('webViewLink')
        except gspread.exceptions.SpreadsheetNotFound:
            logger.info(f"Таблица {sheet_name} не найдена, создается новая")

            file_metadata = {
                'name': sheet_name,
                'mimeType': 'application/vnd.google-apps.spreadsheet',
                'parents': ['1_uGrKw3GJk4vv6STay5J55rcvfV1soIX']
            }
            if folder_id:
                file_metadata['parents'] = [folder_id]

            file = self.drive_service.files().create(
                body=file_metadata,
                fields='id, webViewLink'
            ).execute()

            spreadsheet_id = file.get("id")
            shared_link = file.get("webViewLink")

            # Пауза перед доступом к таблице
            time.sleep(2)
            # Предположим, вы хотите добавить файл в вашу папку (например, 'ваша_папка_id')
            user_folder_id = '1_uGrKw3GJk4vv6STay5J55rcvfV1soIX'

            # Добавим файл в папку пользователя (Мой Диск) и удалим из папки сервиса
            self.drive_service.files().update(
                fileId=spreadsheet_id,
                addParents=user_folder_id,
                removeParents=file_metadata['parents'][0],
                fields='id, parents'
            ).execute()

            print(f"Файл создан и перемещён в папку пользователя: {user_folder_id}")

            # Пауза перед доступом к таблице
            time.sleep(2)

            self.drive_service.permissions().create(
                fileId=spreadsheet_id,
                body={
                    'type': 'anyone',
                    'role': 'writer'
                },
                fields='id'
            ).execute()

            sheet = self.client.open_by_key(spreadsheet_id).sheet1
            logger.info(f"Создана таблица: {sheet_name} (ID: {spreadsheet_id})")
            return sheet, spreadsheet_id, shared_link

    def export_short_summary(self, request, export_data):
        try:
            headers = [
                "ID записи", "Название файла", "Описание документа",
                "Ссылка в исходном облаке", "Ссылка на источник",
                "Формат в базе знаний", "Статус"
            ]
            data = [headers]

            for record in export_data:
                record_full_url = request.build_absolute_uri(record.get_absolute_url())
                record_hyperlink = f'=HYPERLINK("{record_full_url}";"Ссылка на источник")'

                data.append([
                    str(record.id),
                    record.title,
                    record.description or "",
                    record.path,
                    record_hyperlink,
                    "file" if record.output_format == "f" else "text",
                    record.get_status_display()
                ])

            with_retry(lambda: self.short_sheet.clear())
            with_retry(lambda: self.short_sheet.append_rows(data, value_input_option='USER_ENTERED'))
            self._set_data_validation(self.short_sheet, len(export_data))
            self._auto_resize_columns(self.short_sheet, num_columns=7)
            self._set_row_heights(self.short_sheet, num_rows=len(data), pixel_size=21)

            logger.info(f"Успешный экспорт в таблицу {self.short_sheet_name}")
        except Exception as e:
            logger.error(f"Ошибка экспорта: {e}", exc_info=True)
            raise

    def _auto_resize_columns(self, sheet, num_columns):
        spreadsheet_id = sheet.spreadsheet.id
        body = {
            "requests": [{
                "autoResizeDimensions": {
                    "dimensions": {
                        "sheetId": sheet.id,
                        "dimension": "COLUMNS",
                        "startIndex": 0,
                        "endIndex": num_columns
                    }
                }
            }]
        }
        with_retry(lambda: self.sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body
        ).execute())

    def _set_row_heights(self, sheet, num_rows, pixel_size=21):
        spreadsheet_id = sheet.spreadsheet.id
        body = {
            "requests": [{
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet.id,
                        "dimension": "ROWS",
                        "startIndex": 0,
                        "endIndex": num_rows
                    },
                    "properties": {"pixelSize": pixel_size},
                    "fields": "pixelSize"
                }
            }]
        }
        with_retry(lambda: self.sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body
        ).execute())

    def _set_data_validation(self, sheet, num_records):
        spreadsheet_id = sheet.spreadsheet.id
        requests = [{
            "setDataValidation": {
                "range": {
                    "sheetId": sheet.id,
                    "startRowIndex": 1,
                    "endRowIndex": num_records + 1,
                    "startColumnIndex": 5,
                    "endColumnIndex": 6
                },
                "rule": {
                    "condition": {
                        "type": "ONE_OF_LIST",
                        "values": [
                            {"userEnteredValue": "text"},
                            {"userEnteredValue": "file"}
                        ]
                    },
                    "showCustomUi": True,
                    "strict": True
                }
            }
        }]
        body = {"requests": requests}
        with_retry(lambda: self.sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body
        ).execute())

# class GoogleSheetsManager:
#     """
#     Управляет импортом и экспортом данных между базой данных и Google Sheets.
#     """
#     def __init__(self, credentials_file, short_sheet_name="DocScanner_Summary", full_sheet_name="DocScanner_FullSummary"):
#         """
#         Инициализирует менеджер Google Sheets.
#
#         Args:
#             credentials_file (str): Путь к JSON-файлу учетных данных Google API.
#             short_sheet_name (str): Название Google Sheets для краткого экспорта.
#             full_sheet_name (str): Название Google Sheets для полного экспорта.
#         """
#         self.scope = [
#             "https://spreadsheets.google.com/feeds",
#             "https://www.googleapis.com/auth/drive",
#             "https://www.googleapis.com/auth/spreadsheets"
#         ]
#         try:
#             if not os.path.isfile(credentials_file):
#                 logger.error(f"Файл учетных данных не найден: {credentials_file}")
#                 raise FileNotFoundError(f"Credentials file not found: {credentials_file}")
#
#             logger.info(f"Используется файл учетных данных: {credentials_file}")
#             self.credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, self.scope)
#             self.client = gspread.authorize(self.credentials)
#
#             try:
#                 logger.info(f"Попытка открыть таблицу: {short_sheet_name}")
#                 self.short_sheet = self.client.open(short_sheet_name).sheet1
#                 logger.info(f"Таблица {short_sheet_name} успешно открыта")
#             except gspread.exceptions.SpreadsheetNotFound:
#                 logger.info(f"Таблица {short_sheet_name} не найдена, создается новая")
#                 self.short_sheet = self.client.create(short_sheet_name).sheet1
#                 self.short_spreadsheet_id = self.short_sheet.spreadsheet.id
#                 self.short_shared_link = self._share_with_link(self.short_spreadsheet_id)
#                 print(self.short_shared_link)
#                 logger.info(f"Создана новая Google Sheet: {short_sheet_name}")
#             except gspread.exceptions.APIError as e:
#                 logger.error(f"Ошибка API при открытии таблицы {short_sheet_name}: {e}")
#                 raise
#
#             try:
#                 logger.info(f"Попытка открыть таблицу: {full_sheet_name}")
#                 self.full_sheet = self.client.open(full_sheet_name).sheet1
#                 logger.info(f"Таблица {full_sheet_name} успешно открыта")
#             except gspread.exceptions.SpreadsheetNotFound:
#                 logger.info(f"Таблица {full_sheet_name} не найдена, создается новая")
#                 self.full_sheet = self.client.create(full_sheet_name).sheet1
#                 logger.info(f"Создана новая Google Sheet: {full_sheet_name}")
#             except gspread.exceptions.APIError as e:
#                 logger.error(f"Ошибка API при открытии таблицы {full_sheet_name}: {e}")
#                 raise
#
#             self.short_sheet_name = short_sheet_name
#             self.full_sheet_name = full_sheet_name
#             self.short_spreadsheet_id = self.short_sheet.spreadsheet.id
#             self.full_spreadsheet_id = self.full_sheet.spreadsheet.id
#             self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
#             logger.info("Google Sheets Manager успешно инициализирован")
#         except Exception as e:
#             logger.error(f"Ошибка инициализации Google Sheets: {e}", exc_info=True)
#             raise
#
#     def _share_with_link(self, spreadsheet_id):
#         """Делает таблицу доступной для редактирования всем, у кого есть ссылка."""
#         drive_service = build('drive', 'v3', credentials=self.credentials)
#
#         permission = {
#             'type': 'anyone',
#             'role': 'writer',  # можно изменить на 'reader' если нужен только просмотр
#         }
#
#         try:
#             drive_service.permissions().create(
#                 fileId=spreadsheet_id,
#                 body=permission,
#                 fields='id'
#             ).execute()
#             file = drive_service.files().get(fileId=spreadsheet_id, fields='webViewLink').execute()
#             return file.get('webViewLink')
#         except Exception as e:
#             logger.error(f"Не удалось выдать доступ по ссылке: {e}", exc_info=True)
#             return None
#
#
#     def _auto_resize_columns(self, sheet, num_columns):
#         """
#         Автоматически подстраивает ширину столбцов под содержимое.
#         """
#         try:
#             spreadsheet_id = sheet.spreadsheet.id
#             requests = [
#                 {
#                     "autoResizeDimensions": {
#                         "dimensions": {
#                             "sheetId": sheet.id,
#                             "dimension": "COLUMNS",
#                             "startIndex": 0,
#                             "endIndex": num_columns
#                         }
#                     }
#                 }
#             ]
#             body = {"requests": requests}
#             self.sheets_service.spreadsheets().batchUpdate(
#                 spreadsheetId=spreadsheet_id,
#                 body=body
#             ).execute()
#             logger.info(f"Ширина столбцов (0-{num_columns-1}) автоматически настроена для {sheet.spreadsheet.title}")
#         except HttpError as e:
#             logger.error(f"Ошибка настройки ширины столбцов для {sheet.spreadsheet.title}: {e}")
#
#     def _set_row_heights(self, sheet, num_rows, pixel_size=21):
#         """
#         Устанавливает высоту строк для лучшей читаемости.
#         """
#         try:
#             spreadsheet_id = sheet.spreadsheet.id
#             requests = [
#                 {
#                     "updateDimensionProperties": {
#                         "range": {
#                             "sheetId": sheet.id,
#                             "dimension": "ROWS",
#                             "startIndex": 0,
#                             "endIndex": num_rows
#                         },
#                         "properties": {
#                             "pixelSize": pixel_size
#                         },
#                         "fields": "pixelSize"
#                     }
#                 }
#             ]
#             body = {"requests": requests}
#             self.sheets_service.spreadsheets().batchUpdate(
#                 spreadsheetId=spreadsheet_id,
#                 body=body
#             ).execute()
#             logger.info(f"Высота строк (0-{num_rows-1}) установлена в {pixel_size}px для {sheet.spreadsheet.title}")
#         except HttpError as e:
#             logger.error(f"Ошибка настройки высоты строк для {sheet.spreadsheet.title}: {e}")
#
#     def _set_protected_ranges(self, sheet):
#         """
#         Настраивает защиту ячеек, оставляя редактируемыми столбцы D (Название документа) и F (Данные или файл).
#         Разрешает изменение размеров столбцов и строк.
#         """
#         try:
#             spreadsheet_id = sheet.spreadsheet.id
#             requests = [
#                 {
#                     "addProtectedRange": {
#                         "protectedRange": {
#                             "range": {
#                                 "sheetId": sheet.id
#                             },
#                             "description": "Protect entire sheet except columns D and F",
#                             "warningOnly": False,
#                             "editors": {
#                                 "users": [self.credentials.service_account_email]
#                             },
#                             "unprotectedRanges": [
#                                 {
#                                     "sheetId": sheet.id,
#                                     "startRowIndex": 1,
#                                     "startColumnIndex": 3,  # Столбец D
#                                     "endColumnIndex": 4
#                                 },
#                                 {
#                                     "sheetId": sheet.id,
#                                     "startRowIndex": 1,
#                                     "startColumnIndex": 5,  # Столбец F
#                                     "endColumnIndex": 6
#                                 }
#                             ]
#                         }
#                     }
#                 }
#             ]
#             body = {"requests": requests}
#             self.sheets_service.spreadsheets().batchUpdate(
#                 spreadsheetId=spreadsheet_id,
#                 body=body
#             ).execute()
#             logger.info(f"Защита ячеек настроена для {sheet.spreadsheet.title}: редактируемые столбцы D и F, разрешено изменение размеров")
#         except HttpError as e:
#             logger.error(f"Ошибка настройки защиты ячеек для {sheet.spreadsheet.title}: {e}")
#             raise
#
#     def _set_data_validation(self, sheet, records):
#         """
#         Настраивает поле выбора для столбца F (Данные или файл) с значениями text/file.
#         """
#         try:
#             spreadsheet_id = sheet.spreadsheet.id
#             requests = []
#             for i, _ in enumerate(records, start=2):
#                 requests.append({
#                     "setDataValidation": {
#                         "range": {
#                             "sheetId": sheet.id,
#                             "startRowIndex": i-1,
#                             "endRowIndex": i,
#                             "startColumnIndex": 5,  # Столбец F
#                             "endColumnIndex": 6
#                         },
#                         "rule": {
#                             "condition": {
#                                 "type": "ONE_OF_LIST",
#                                 "values": [
#                                     {"userEnteredValue": "text"},
#                                     {"userEnteredValue": "file"}
#                                 ]
#                             },
#                             "showCustomUi": True,
#                             "strict": True
#                         }
#                     }
#                 })
#             if requests:
#                 body = {"requests": requests}
#                 self.sheets_service.spreadsheets().batchUpdate(
#                     spreadsheetId=spreadsheet_id,
#                     body=body
#                 ).execute()
#                 logger.info(f"Поле выбора настроено для столбца 'Данные' в {sheet.spreadsheet.title}")
#         except HttpError as e:
#             logger.error(f"Ошибка настройки поля заполнения для {sheet.spreadsheet.title}: {e}")
#
#     def export_short_summary(self, request, export_data):
#         """
#         Экспортирует данные NetworkDocuments в Google Sheet (DocScanner_Summary).
#         """
#
#         try:
#             headers = [
#                 "ID записи",
#                 "Название файла",
#                 "Описание документа",
#                 "Ссылка в исходном облаке",
#                 "Ссылка на источник",
#                 "Формат в базе знаний",
#                 "Статус"
#             ]
#             data = [headers]
#             for record in export_data:
#                 record_full_url = request.build_absolute_uri(record.get_absolute_url())
#                 record_hyperlink = f'=HYPERLINK("{record_full_url}";"Ссылка на источник")'
#
#                 data.append([
#                     str(record.id),
#                     record.title,
#                     record.description if record.description else "",
#                     record.path,
#                     record_hyperlink,
#                     "file" if record.output_format == "f" else "text",
#                     record.get_status_display()
#                 ])
#
#             self.short_sheet.clear()
#             self.short_sheet.append_rows(data, value_input_option='USER_ENTERED')
#             self._set_data_validation(self.short_sheet, export_data)
#             # self.set_protected_ranges(self.short_sheet)
#             self._auto_resize_columns(self.short_sheet, num_columns=7)
#             self._set_row_heights(self.short_sheet, num_rows=len(data), pixel_size=21)
#             logger.info(f"Successfully exported short summary to: {self.short_sheet_name}")
#         except gspread.exceptions.APIError as e:
#             logger.error(f"Google Sheets API error during export: {e}")
#         except Exception as e:
#             logger.error(f"Error during export: {e}")
#             raise
#
#
#     def import_short_summary(self):
#         """
#         Импортирует краткий свод из Google Sheet и обновляет базу данных.
#         """
#         session = SessionLocal()
#         try:
#             data = self.short_sheet.get_all_values()
#             if not data or len(data) < 2:
#                 logger.info("Google Sheets пустой или не содержит данных.")
#                 return
#
#             headers = data[0]
#             expected_headers = [
#                 "ID записи",
#                 "Название файла",
#                 "Название документа",
#                 "Название папки",
#                 "Ссылка на",
#                 "Данные или",
#                 "Статус"
#             ]
#             if headers != expected_headers:
#                 logger.error(f"Неверный формат заголовков: {headers}")
#                 return
#
#             for row in data[1:]:
#                 record_id, name, document_name, _, _, data_type, _ = row
#                 try:
#                     record_id = int(record_id) if record_id.strip() else None
#                 except ValueError:
#                     logger.warning(f"Invalid record ID: {record_id}")
#                     record_id = None
#
#                 if data_type not in ["text", "file"]:
#                     logger.warning(f"Invalid data_type: {data_type}, using 'file'")
#                     data_type = "file"
#
#                 file_record = None
#                 if record_id:
#                     file_record = session.query(FileRecord).filter_by(id=record_id).first()
#
#                 if file_record and file_record.name == name:
#                     updated = False
#                     if file_record.data_type.value != data_type:
#                         file_record.data_type = DataType[data_type]
#                         updated = True
#                     if document_name.strip() and file_record.document_name != document_name:
#                         file_record.document_name.data = document_name[:500]
#                         updated = True
#                     if updated:
#                         logger.info(f"Updated record ID={record_id}: name={name}")
#                 else:
#                     file_record = session.query(FileRecord).filter_by(name=name).first()
#                     if file_record:
#                         updated = False
#                         if file_record.data_type.value != data_type:
#                             file_record.data_type = DataType[data_type]
#                             updated = True
#                         if document_name.strip() and file_record.document_name != document_name:
#                             file_record.data_type.data = document_name[:500]
#                             updated = True
#                         if updated:
#                             logger.info(f"Updated record by name: name={name}")
#                     else:
#                         logger.warning(f"Skipped new record: name={name}, no URL provided")
#
#                 session.commit()
#             logger.info(f"Import completed: {self.short_sheet_name}")
#         except gspread.exceptions.APIError as e:
#             logger.error(f"Google Sheets API error during import: {e}")
#             session.rollback()
#         except Exception as e:
#             logger.error(f"Error during import: {e}")
#             session.rollback()
#         finally:
#             session.close()
#
#     def export_full_summary(self, preview_chars=200):
#         """
#         Экспортирует полный свод данных в Google Sheet (DocScanner_FullSummary).
#         """
#         session = SessionLocal()
#         try:
#             file_records = session.query(FileRecord).order_by(FileRecord.name).all()
#             if not file_records:
#                 logger.info("Нет записей для экспорта в полный свод.")
#                 return
#
#             headers = [
#                 "ID записи",
#                 "Название файла",
#                 "Ссылка на файл",
#                 "Путь",
#                 "SHA-512",
#                 "Размер (байт)",
#                 "Последнее изменение",
#                 "ETag",
#                 "Дата создания",
#                 "Статус удаления",
#                 "Тип данных",
#                 "Название документа",
#                 "ID версии",
#                 "ID файла (версия)",
#                 "Путь к тексту",
#                 "Дата обработки",
#                 "Метод обработки",
#                 "Отчет качества",
#                 "Превью текста"
#             ]
#             data = [headers]
#             for file in file_records:
#                 latest = file.latest_version
#                 folder_name, folder_url = self.get_parent_folder_info(file.path, file.id)
#                 deleted_at = str(file.soft_deleted_at) if file.soft_deleted_at else ""
#                 preview = ""
#                 if latest and latest.text_path and os.path.isfile(latest.text_path):
#                     try:
#                         with open(latest.text_path, 'r', encoding='utf-8') as f:
#                             preview = f.read(preview_chars).strip().replace("\n", " ")
#                             if len(preview) == preview_chars:
#                                 preview += "..."
#                     except Exception as e:
#                         preview = f"[Text read error: {e}]"
#                 else:
#                     preview = "[File not found]"
#
#                 data.append([
#                     str(file.id),
#                     file.name,
#                     file.url,
#                     file.path,
#                     file.sha512,
#                     str(file.size),
#                     str(file.last_modified),
#                     file.etag or "",
#                     str(file.created_at),
#                     deleted_at,
#                     file.data_type.value,
#                     file.document_name or "",
#                     str(latest.id) if latest else "",
#                     str(latest.file_id) if latest else "",
#                     latest.text_path or "",
#                     str(latest.processed_at) if latest else "",
#                     latest.method or "",
#                     latest.quality_report or "",
#                     preview
#                 ])
#
#             self.full_sheet.clear()
#             self.full_sheet.append_rows(data)
#             self.auto_resize_columns(self.full_sheet, num_columns=19)
#             self.set_row_heights(self.full_sheet, num_rows=len(data))
#             logger.info(f"Successfully exported full summary to: {self.full_sheet_name}")
#         except gspread.exceptions.APIError as e:
#             logger.error(f"Google Sheets API error during export_full: {e}")
#             raise
#         except Exception as e:
#             logger.error(f"Error during export_full: {e}")
#             raise
#         finally:
#             session.close()



if __name__ == "__main__":
    credentials_path = os.path.join(project_root, "credentials.json")
    sheets_manager = GoogleSheetsManager(credentials_path)
    sheets_manager.export_short_summary()
    # sheets_manager.export_full_summary()