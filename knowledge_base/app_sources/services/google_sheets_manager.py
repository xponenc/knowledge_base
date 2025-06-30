import os
import sys
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

class GoogleSheetsManager:
    """
    Управляет импортом и экспортом данных между базой данных и Google Sheets.
    """
    def __init__(self, credentials_file, short_sheet_name="DocScanner_Summary", full_sheet_name="DocScanner_FullSummary"):
        """
        Инициализирует менеджер Google Sheets.

        Args:
            credentials_file (str): Путь к JSON-файлу учетных данных Google API.
            short_sheet_name (str): Название Google Sheets для краткого экспорта.
            full_sheet_name (str): Название Google Sheets для полного экспорта.
        """
        self.scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets"
        ]
        try:
            if not os.path.isfile(credentials_file):
                logger.error(f"Файл учетных данных не найден: {credentials_file}")
                raise FileNotFoundError(f"Credentials file not found: {credentials_file}")

            logger.info(f"Используется файл учетных данных: {credentials_file}")
            self.credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, self.scope)
            self.client = gspread.authorize(self.credentials)

            try:
                logger.info(f"Попытка открыть таблицу: {short_sheet_name}")
                self.short_sheet = self.client.open(short_sheet_name).sheet1
                logger.info(f"Таблица {short_sheet_name} успешно открыта")
            except gspread.exceptions.SpreadsheetNotFound:
                logger.info(f"Таблица {short_sheet_name} не найдена, создается новая")
                self.short_sheet = self.client.create(short_sheet_name).sheet1
                logger.info(f"Создана новая Google Sheet: {short_sheet_name}")
            except gspread.exceptions.APIError as e:
                logger.error(f"Ошибка API при открытии таблицы {short_sheet_name}: {e}")
                raise

            try:
                logger.info(f"Попытка открыть таблицу: {full_sheet_name}")
                self.full_sheet = self.client.open(full_sheet_name).sheet1
                logger.info(f"Таблица {full_sheet_name} успешно открыта")
            except gspread.exceptions.SpreadsheetNotFound:
                logger.info(f"Таблица {full_sheet_name} не найдена, создается новая")
                self.full_sheet = self.client.create(full_sheet_name).sheet1
                logger.info(f"Создана новая Google Sheet: {full_sheet_name}")
            except gspread.exceptions.APIError as e:
                logger.error(f"Ошибка API при открытии таблицы {full_sheet_name}: {e}")
                raise

            self.short_sheet_name = short_sheet_name
            self.full_sheet_name = full_sheet_name
            self.short_spreadsheet_id = self.short_sheet.spreadsheet.id
            self.full_spreadsheet_id = self.full_sheet.spreadsheet.id
            self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
            logger.info("Google Sheets Manager успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации Google Sheets: {e}", exc_info=True)
            raise

    # def _get_parent_folder_info(self, file_path, file_url):
    #     """
    #     Извлекает название и ссылку на родительскую папку из пути и URL файла.
    #     """
    #     parsed_path = file_path.strip("/").split("/")
    #     if len(parsed_path) > 1:
    #         folder_name = parsed_path[-2]
    #         folder_url = urljoin(WEBDAV_BASE_URL, "/".join(parsed_path[:-1]) + "/")
    #     else:
    #         folder_name = "/"
    #         folder_url = WEBDAV_BASE_URL
    #     return folder_name, folder_url

    def _auto_resize_columns(self, sheet, num_columns):
        """
        Автоматически подстраивает ширину столбцов под содержимое.
        """
        try:
            spreadsheet_id = sheet.spreadsheet.id
            requests = [
                {
                    "autoResizeDimensions": {
                        "dimensions": {
                            "sheetId": sheet.id,
                            "dimension": "COLUMNS",
                            "startIndex": 0,
                            "endIndex": num_columns
                        }
                    }
                }
            ]
            body = {"requests": requests}
            self.sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=body
            ).execute()
            logger.info(f"Ширина столбцов (0-{num_columns-1}) автоматически настроена для {sheet.spreadsheet.title}")
        except HttpError as e:
            logger.error(f"Ошибка настройки ширины столбцов для {sheet.spreadsheet.title}: {e}")

    def _set_row_heights(self, sheet, num_rows, pixel_size=21):
        """
        Устанавливает высоту строк для лучшей читаемости.
        """
        try:
            spreadsheet_id = sheet.spreadsheet.id
            requests = [
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet.id,
                            "dimension": "ROWS",
                            "startIndex": 0,
                            "endIndex": num_rows
                        },
                        "properties": {
                            "pixelSize": pixel_size
                        },
                        "fields": "pixelSize"
                    }
                }
            ]
            body = {"requests": requests}
            self.sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=body
            ).execute()
            logger.info(f"Высота строк (0-{num_rows-1}) установлена в {pixel_size}px для {sheet.spreadsheet.title}")
        except HttpError as e:
            logger.error(f"Ошибка настройки высоты строк для {sheet.spreadsheet.title}: {e}")

    def _set_protected_ranges(self, sheet):
        """
        Настраивает защиту ячеек, оставляя редактируемыми столбцы D (Название документа) и F (Данные или файл).
        Разрешает изменение размеров столбцов и строк.
        """
        try:
            spreadsheet_id = sheet.spreadsheet.id
            requests = [
                {
                    "addProtectedRange": {
                        "protectedRange": {
                            "range": {
                                "sheetId": sheet.id
                            },
                            "description": "Protect entire sheet except columns D and F",
                            "warningOnly": False,
                            "editors": {
                                "users": [self.credentials.service_account_email]
                            },
                            "unprotectedRanges": [
                                {
                                    "sheetId": sheet.id,
                                    "startRowIndex": 1,
                                    "startColumnIndex": 3,  # Столбец D
                                    "endColumnIndex": 4
                                },
                                {
                                    "sheetId": sheet.id,
                                    "startRowIndex": 1,
                                    "startColumnIndex": 5,  # Столбец F
                                    "endColumnIndex": 6
                                }
                            ]
                        }
                    }
                }
            ]
            body = {"requests": requests}
            self.sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=body
            ).execute()
            logger.info(f"Защита ячеек настроена для {sheet.spreadsheet.title}: редактируемые столбцы D и F, разрешено изменение размеров")
        except HttpError as e:
            logger.error(f"Ошибка настройки защиты ячеек для {sheet.spreadsheet.title}: {e}")
            raise

    def _set_data_validation(self, sheet, records):
        """
        Настраивает поле выбора для столбца F (Данные или файл) с значениями text/file.
        """
        try:
            spreadsheet_id = sheet.spreadsheet.id
            requests = []
            for i, _ in enumerate(records, start=2):
                requests.append({
                    "setDataValidation": {
                        "range": {
                            "sheetId": sheet.id,
                            "startRowIndex": i-1,
                            "endRowIndex": i,
                            "startColumnIndex": 5,  # Столбец F
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
                })
            if requests:
                body = {"requests": requests}
                self.sheets_service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body=body
                ).execute()
                logger.info(f"Поле выбора настроено для столбца 'Данные' в {sheet.spreadsheet.title}")
        except HttpError as e:
            logger.error(f"Ошибка настройки поля заполнения для {sheet.spreadsheet.title}: {e}")

    def export_short_summary(self, export_data):
        """
        Экспортирует данные NetworkDocuments в Google Sheet (DocScanner_Summary).
        """

        try:
            headers = [
                "ID записи",
                "Название файла",
                "Название документа",
                "Название папки",
                "Ссылка на",
                "Данные или",
                "Статус"
            ]
            data = [headers]
            for record in export_data:
                # folder_name, folder_url = self.get_parent_folder_info(record.path, record.id)

                # folder_path = "/".join(record.path.strip("/").split("/")[:-1])  # Убираем имя файла из пути
                # folder_url = f"https://cloud.academydpo.org/s/{WEBDAV_TOKEN}?dir=/{quote(folder_path)}"
                folder_hyperlink = f'=HYPERLINK("{record.url}";"Ссылка на папку")'

                deleted_at = str(record.soft_deleted_at) if record.soft_deleted_at else ""
                data.append([
                    str(record.id),
                    record.title,
                    record.document_name or "",
                    record.path,
                    folder_hyperlink,
                    record.data_type.value,
                    deleted_at
                ])

            self.short_sheet.clear()
            self.short_sheet.append_rows(data, value_input_option='USER_ENTERED')
            self._set_data_validation(self.short_sheet, export_data)
            # self.set_protected_ranges(self.short_sheet)
            self._auto_resize_columns(self.short_sheet, num_columns=7)
            self._set_row_heights(self.short_sheet, num_rows=len(data), pixel_size=21)
            logger.info(f"Successfully exported short summary to: {self.short_sheet_name}")
        except gspread.exceptions.APIError as e:
            logger.error(f"Google Sheets API error during export: {e}")
        except Exception as e:
            logger.error(f"Error during export: {e}")
            raise


    def import_short_summary(self):
        """
        Импортирует краткий свод из Google Sheet и обновляет базу данных.
        """
        session = SessionLocal()
        try:
            data = self.short_sheet.get_all_values()
            if not data or len(data) < 2:
                logger.info("Google Sheets пустой или не содержит данных.")
                return

            headers = data[0]
            expected_headers = [
                "ID записи",
                "Название файла",
                "Название документа",
                "Название папки",
                "Ссылка на",
                "Данные или",
                "Статус"
            ]
            if headers != expected_headers:
                logger.error(f"Неверный формат заголовков: {headers}")
                return

            for row in data[1:]:
                record_id, name, document_name, _, _, data_type, _ = row
                try:
                    record_id = int(record_id) if record_id.strip() else None
                except ValueError:
                    logger.warning(f"Invalid record ID: {record_id}")
                    record_id = None

                if data_type not in ["text", "file"]:
                    logger.warning(f"Invalid data_type: {data_type}, using 'file'")
                    data_type = "file"

                file_record = None
                if record_id:
                    file_record = session.query(FileRecord).filter_by(id=record_id).first()

                if file_record and file_record.name == name:
                    updated = False
                    if file_record.data_type.value != data_type:
                        file_record.data_type = DataType[data_type]
                        updated = True
                    if document_name.strip() and file_record.document_name != document_name:
                        file_record.document_name.data = document_name[:500]
                        updated = True
                    if updated:
                        logger.info(f"Updated record ID={record_id}: name={name}")
                else:
                    file_record = session.query(FileRecord).filter_by(name=name).first()
                    if file_record:
                        updated = False
                        if file_record.data_type.value != data_type:
                            file_record.data_type = DataType[data_type]
                            updated = True
                        if document_name.strip() and file_record.document_name != document_name:
                            file_record.data_type.data = document_name[:500]
                            updated = True
                        if updated:
                            logger.info(f"Updated record by name: name={name}")
                    else:
                        logger.warning(f"Skipped new record: name={name}, no URL provided")

                session.commit()
            logger.info(f"Import completed: {self.short_sheet_name}")
        except gspread.exceptions.APIError as e:
            logger.error(f"Google Sheets API error during import: {e}")
            session.rollback()
        except Exception as e:
            logger.error(f"Error during import: {e}")
            session.rollback()
        finally:
            session.close()

    def export_full_summary(self, preview_chars=200):
        """
        Экспортирует полный свод данных в Google Sheet (DocScanner_FullSummary).
        """
        session = SessionLocal()
        try:
            file_records = session.query(FileRecord).order_by(FileRecord.name).all()
            if not file_records:
                logger.info("Нет записей для экспорта в полный свод.")
                return

            headers = [
                "ID записи",
                "Название файла",
                "Ссылка на файл",
                "Путь",
                "SHA-512",
                "Размер (байт)",
                "Последнее изменение",
                "ETag",
                "Дата создания",
                "Статус удаления",
                "Тип данных",
                "Название документа",
                "ID версии",
                "ID файла (версия)",
                "Путь к тексту",
                "Дата обработки",
                "Метод обработки",
                "Отчет качества",
                "Превью текста"
            ]
            data = [headers]
            for file in file_records:
                latest = file.latest_version
                folder_name, folder_url = self.get_parent_folder_info(file.path, file.id)
                deleted_at = str(file.soft_deleted_at) if file.soft_deleted_at else ""
                preview = ""
                if latest and latest.text_path and os.path.isfile(latest.text_path):
                    try:
                        with open(latest.text_path, 'r', encoding='utf-8') as f:
                            preview = f.read(preview_chars).strip().replace("\n", " ")
                            if len(preview) == preview_chars:
                                preview += "..."
                    except Exception as e:
                        preview = f"[Text read error: {e}]"
                else:
                    preview = "[File not found]"

                data.append([
                    str(file.id),
                    file.name,
                    file.url,
                    file.path,
                    file.sha512,
                    str(file.size),
                    str(file.last_modified),
                    file.etag or "",
                    str(file.created_at),
                    deleted_at,
                    file.data_type.value,
                    file.document_name or "",
                    str(latest.id) if latest else "",
                    str(latest.file_id) if latest else "",
                    latest.text_path or "",
                    str(latest.processed_at) if latest else "",
                    latest.method or "",
                    latest.quality_report or "",
                    preview
                ])

            self.full_sheet.clear()
            self.full_sheet.append_rows(data)
            self.auto_resize_columns(self.full_sheet, num_columns=19)
            self.set_row_heights(self.full_sheet, num_rows=len(data))
            logger.info(f"Successfully exported full summary to: {self.full_sheet_name}")
        except gspread.exceptions.APIError as e:
            logger.error(f"Google Sheets API error during export_full: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during export_full: {e}")
            raise
        finally:
            session.close()



if __name__ == "__main__":
    credentials_path = os.path.join(project_root, "credentials.json")
    sheets_manager = GoogleSheetsManager(credentials_path)
    sheets_manager.export_short_summary()
    # sheets_manager.export_full_summary()