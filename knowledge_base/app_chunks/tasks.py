from typing import List, Dict, Type

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from app_chunks.models import Chunk, ChunkStatus
from app_chunks.splitters.base import BaseSplitter
from app_sources.content_models import RawContent, CleanedContent
from app_sources.source_models import OutputDataType
from utils.setup_logger import setup_logger

chunk_logger = setup_logger(name="chunk_logger", log_dir="logs", log_file="chunking_debug.log")


@shared_task(bind=True)
def bulk_chunks_create(
        self,
        content_list: List,
        report_pk: int,
        splitter_cls: Type[BaseSplitter],
        splitter_config: Dict,
        author_pk: int,
) -> str:
    """
    Разбивает веб-страницы на чанки с использованием заданного сплиттера и сохраняет их в базу данных.

    :param self: Celery-задача (для отображения прогресса).
    :param content_list: Список объектов URLContent для обработки.
    :param report_pk: Первичный ключ отчёта обновления.
    :param splitter_cls: Класс-наследник BaseSplitter.
    :param splitter_config: Словарь с конфигурацией для объекта класса-наследника BaseSplitter.
    :param author_pk: Идентификатор пользователя, создавшего чанки.
    :return: Строка с информацией о завершении задачи.
    """
    progress_recorder = ProgressRecorder(self)
    total = len(content_list)
    progress_description_base = f"Создание чанков для {total} веб-страниц"
    progress_step = max(1, total // 100)

    progress_recorder.set_progress(0, total, description=progress_description_base)
    chunk_logger.info(f"[START] Разбиение {total} элементов. Отчёт ID={report_pk}")

    bulk_container = []
    batch_size = 900

    splitter = splitter_cls(splitter_config)

    for i, content in enumerate(content_list):
        try:
            body = content.body
            metadata = content.metadata or {}
            url = content.url.url

            metadata["url"] = url

            documents = splitter.split(metadata=metadata, text_to_split=body)

            for doc_num, document in enumerate(documents):
                chunk = Chunk(
                    url_content=content,
                    status=ChunkStatus.READY.value,
                    report_id=report_pk,
                    metadata=document.metadata,
                    page_content=document.page_content,
                    splitter_cls=splitter.__class__.__name__,
                    splitter_config=splitter.config,
                    author_id=author_pk,
                )
                bulk_container.append(chunk)
                chunk_logger.debug(
                    f"[CHUNK CREATED] URLContent ID={content.id}, URL='{url}', Чанк #{doc_num + 1} из {len(documents)}"
                )

        except Exception as e:
            chunk_logger.exception(
                f"[ERROR] Не удалось создать чанки для URLContent ID={getattr(content, 'id', '?')}, "
                f"URL='{getattr(content.url, 'url', '?')}', index={i}. Ошибка: {e}"
            )
            continue

        if (i + 1) % progress_step == 0 or i + 1 == total:
            progress_recorder.set_progress(i + 1, total, description=progress_description_base)

        if len(bulk_container) >= batch_size:
            Chunk.objects.bulk_create(bulk_container)
            chunk_logger.info(f"[BULK SAVE] Сохранено {len(bulk_container)} чанков на итерации {i}")
            bulk_container.clear()

    if bulk_container:
        Chunk.objects.bulk_create(bulk_container)
        chunk_logger.info(f"[FINAL BULK SAVE] Сохранено {len(bulk_container)} чанков в финальной партии.")

    chunk_logger.info(f"[COMPLETE] Обработка завершена. Всего обработано: {total}")
    progress_recorder.set_progress(total, total, description="Генерация чанков завершена")

    return f"Успешно создано чанков для {total} URLContent."


@shared_task(bind=True)
def universal_bulk_chunks_create(
        self,
        sources_list: List,
        report_pk: int,
        splitter_cls: Type[BaseSplitter],
        splitter_config: Dict,
        author_pk: int,
) -> str:
    """
    Разбивает веб-страницы на чанки с использованием заданного сплиттера и сохраняет их в базу данных.

    :param self: Celery-задача (для отображения прогресса).
    :param sources_list: Список объектов класса NetworkDocument или LocalDocument.
    :param report_pk: Первичный ключ отчёта обновления объекта класса ChunkingReport.
    :param splitter_cls: Класс-наследник BaseSplitter.
    :param splitter_config: Словарь с конфигурацией для объекта класса-наследника BaseSplitter.
    :param author_pk: Идентификатор пользователя, создавшего чанки.
    :return: Строка с информацией о завершении задачи.
    """

    progress_recorder = ProgressRecorder(self)
    total = len(sources_list)
    progress_description_base = f"Создание чанков для {total} документов"
    progress_step = max(1, total // 100)

    progress_recorder.set_progress(0, total, description=progress_description_base)
    chunk_logger.info(f"[START] Разбиение {total} элементов. Отчёт ID={report_pk}")

    bulk_container = []
    batch_size = 500

    splitter = splitter_cls(splitter_config)

    for i, source in enumerate(sources_list):
        try:
            output_format = source.output_format
            if output_format == OutputDataType.file.value:
                # Выходной формат документа в БД файл
                content = source.active_raw_content

                body = f" {source.title}" if source.title else ""
                body += f" {source.description}" if source.description else ""
                body += f" [Ссылка на файл]{source.file_get_url}"

            else:
                content = getattr(source.active_raw_content, "cleanedcontent")
                if content is None:
                    chunk_logger.exception(
                        f"[ERROR] Не удалось создать чанки для {source.__class__.__name__} "
                        f"ID={getattr(source, 'id', '?')}, "
                        f" задан формат вывода в чанки '{output_format}',"
                        f" для RawContent ID={getattr(content, 'id', '?')} не найден CleanedContent"
                    )
                    continue

                body = content.file.read().decode("utf-8")

            metadata = {
                "output_format": "file" if source.output_format == OutputDataType.file.value else "text",
                "file_get_url": source.file_get_url,
                "tags": source.tags,
                "file_name": source.title,
            }

            documents = splitter.split(metadata=metadata, text_to_split=body)

            for doc_num, document in enumerate(documents):
                chunk = Chunk(
                    status=ChunkStatus.READY.value,
                    report_id=report_pk,
                    metadata=document.metadata,
                    page_content=document.page_content,
                    splitter_cls=splitter.__class__.__name__,
                    splitter_config=splitter.config,
                    author_id=author_pk,
                )
                if isinstance(content, RawContent):
                    chunk.raw_content = content
                elif isinstance(content, CleanedContent):
                    chunk.cleaned_content = content
                else:
                    chunk_logger.exception(
                        f"Не удалось создать чанк для {source.__class__.__name__} "
                        f"ID={getattr(source, 'id', '?')}, "
                        f" неизвестный класс контента: {type(content).__name__}"
                    )
                    continue
                bulk_container.append(chunk)
                chunk_logger.debug(
                    f"[CHUNK CREATED] {source.__class__.__name__}  ID={getattr(source, 'id', '?')}"
                    f" {content.__class__.__name__} ID={getattr(content, 'id', '?')},"
                    f" Чанк #{doc_num + 1} из {len(documents)}"
                )

        except Exception as e:
            chunk_logger.exception(
                f"[ERROR] Не удалось создать чанки для {source.__class__.__name__} "
                f"ID={getattr(source, 'id', '?')}, index={i}. Ошибка: {e}"
            )
            continue

        if (i + 1) % progress_step == 0 or i + 1 == total:
            progress_recorder.set_progress(i + 1, total, description=progress_description_base)

        if len(bulk_container) >= batch_size:
            Chunk.objects.bulk_create(bulk_container)
            chunk_logger.info(f"[BULK SAVE] Сохранено {len(bulk_container)} чанков на итерации {i}")
            bulk_container.clear()

    if bulk_container:
        Chunk.objects.bulk_create(bulk_container)
        chunk_logger.info(f"[FINAL BULK SAVE] Сохранено {len(bulk_container)} чанков в финальной партии.")

    chunk_logger.info(f"[COMPLETE] Обработка завершена. Всего обработано: {total}")
    progress_recorder.set_progress(total, total, description="Генерация чанков завершена")

    return f"Успешно создано чанков для {total} URLContent."
