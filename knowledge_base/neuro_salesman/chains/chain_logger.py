import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from utils.setup_logger import setup_logger

logger = setup_logger(name=__name__, log_dir="logs/neuro_salesman", log_file="ns.log")


class ChainLogger:
    """
    Унифицированный логгер для цепочек (Runnable/LLM).

    Поддерживает разные уровни логирования (info, debug, warning, error) и debug-mode:
    - В debug_mode все сообщения дополнительно выводятся в консоль.
    - exc_info можно передавать для включения трассировки исключений в лог.

    Args:
        prefix (str): префикс для всех сообщений логгера (например, "[Greeting Extractor]").
        debug_mode (bool): если True — вывод сообщений в консоль помимо логгера.
    """
    def __init__(self, prefix="Chain"):
        self.prefix = prefix

    def log(self, session_info: str, level: str, message: str, exc: Exception = None) -> None:
        """
        Логирует сообщение на заданном уровне с включением префикса и session_info прямо в текст.
        """
        final_message = f"[{session_info}] [{self.prefix}] {message}"

        log_func = getattr(logger, level, None)
        if log_func is None:
            raise ValueError(f"Invalid log level: {level}")

        log_func(final_message, exc_info=exc, stacklevel=2)

