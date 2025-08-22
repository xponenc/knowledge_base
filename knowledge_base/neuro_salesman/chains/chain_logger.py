import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print(sys.path)

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
    def __init__(self, prefix="[Chain]", debug_mode=False):
        self.prefix = prefix
        self.debug_mode = debug_mode

    def log(self, session_info, level, message, exc: Exception = None):
        formatted = f"[{self.prefix}][{session_info}] {message}"
        if self.debug_mode:
            print(formatted)

        # stacklevel=2 чтобы показывалось место вызова, а не chain_logger.py
        kwargs = {"exc_info": exc, "stacklevel": 2}
        if level == "info":
            logger.info(formatted, **kwargs)
        elif level == "debug":
            logger.debug(formatted, **kwargs)
        elif level == "warning":
            logger.warning(formatted, **kwargs)
        elif level == "error":
            logger.error(formatted, **kwargs)