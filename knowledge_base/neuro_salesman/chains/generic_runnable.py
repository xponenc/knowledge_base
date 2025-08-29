import time
from copy import copy
from typing import Callable, Dict, Any
from langchain_core.runnables import Runnable
from .chain_logger import ChainLogger
from ..config import EMPTY_MESSAGE


class GenericRunnable(Runnable):
    """
    Универсальная обертка для любых цепочек (Runnable, LLM и т.п.).

    Позволяет:
    - Подготавливать входные данные через функцию `input_mapping`;
    - Вызывать вложенную цепочку (`chain`);
    - Преобразовывать результат в удобный формат через `output_mapping`;
    - Вести унифицированное логирование через `ChainLogger`.

    Это заменяет узкоспециализированные классы вроде KeyedRunnable,
    RouterRunnable и т.п., сохраняя при этом гибкость.

    Args:
        chain (Runnable):
            Цепочка или LLM, которую нужно вызвать.
        output_key (str):
            Ключ, под которым будет сохранен результат в выходном словаре.
            Если `output_mapping` переопределён — этот ключ можно игнорировать.
        prefix (str, optional):
            Префикс для логов (например, "Greeting Extractor").

        input_mapping (Callable[[Dict[str, Any]], Dict[str, Any]], optional):
            Функция для подготовки входных данных для `chain`.
            По умолчанию возвращает inputs как есть.
        output_mapping (Callable[[Any, Dict[str, Any]], Dict[str, Any]], optional):
            Функция для преобразования результата.
            По умолчанию добавляет результат в inputs под ключом `output_key`.

    Example:
        >>> runnable = GenericRunnable(
        ...     chain=llm_chain,
        ...     output_key="greeting",
        ...     prefix="Greeting Extractor",
        ...     input_mapping=lambda inputs: {"text": inputs["last_message_from_client"]},
        ...     output_mapping=lambda result, inputs: {**inputs, "greeting": result}
        ... )
        >>> result = runnable.invoke({"last_message_from_client": "Привет"})
        >>> print(result["greeting"])
        "Здравствуйте!"
    """

    def __init__(
        self,
        chain: Runnable,
        output_key: str,
        prefix: str = "[Chain]",
        input_mapping: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
        output_mapping: Callable[[Any, Dict[str, Any]], Dict[str, Any]] = None,
    ):
        self.chain = chain
        self.output_key = output_key
        self.logger = ChainLogger(prefix=prefix)
        self.input_mapping = input_mapping or (lambda inputs: inputs)
        self.output_mapping = output_mapping or (lambda result, inputs: {**inputs, self.output_key: result})

    def invoke(self, inputs: Dict[str, Any], config=None, **kwargs) -> Dict[str, Any]:
        """
        Запускает вложенную цепочку с подготовкой входов и пост-обработкой результата.

        Args:
            inputs (Dict[str, Any]): Словарь входных данных.
            config (Any, optional): Конфигурация выполнения (например, параметры LLM).
            **kwargs: Дополнительные аргументы, передаваемые во вложенный chain.

        Returns:
            Dict[str, Any]: Обновленный словарь входов с добавленным результатом.
        """

        start_time = time.monotonic()

        updates = {}
        for key in ("original_inputs", "report_and_router", "final_result"):
            key_inputs = inputs.get(key)
            if key_inputs and isinstance(key_inputs, dict):
                updates.update(key_inputs)

        for key in ("original_inputs", "report_and_router", "final_result"):
            inputs.pop(key, None)

        inputs.update(updates)

        session_info = f"{inputs.get('session_type', 'n/a')}:{inputs.get('session_id', 'n/a')}"
        self.logger.log(session_info, "debug", f"inputs: {inputs}")
        try:
            # Подготовка входов
            mapped_inputs = self.input_mapping(inputs)
            self.logger.log(session_info, "debug", f"mapped_inputs={mapped_inputs}")
            # Вызов модели/цепочки
            result = self.chain.invoke(mapped_inputs, config=config, **kwargs)
            elapsed = time.monotonic() - start_time
            self.logger.log(session_info, "info", f"finished in {elapsed:.2f}s")
            self.logger.log(session_info, "debug", f"raw_result={result}")

            # Пост-обработка результата
            final_output = self.output_mapping(result, inputs)
            return final_output

        except Exception as e:
            self.logger.log(session_info, "error", f"Ошибка: {str(e)}", exc=e)
            return self.output_mapping(EMPTY_MESSAGE, inputs)
