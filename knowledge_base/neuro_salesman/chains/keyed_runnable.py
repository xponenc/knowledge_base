import time

from langchain_core.runnables import Runnable

from neuro_salesman.chains.chain_logger import ChainLogger
from neuro_salesman.config import EMPTY_MESSAGE


class KeyedRunnable(Runnable):
    """
    Универсальная обертка для цепочек, возвращающая результат под ключом output_key.
    Использует ChainLogger для унифицированного логирования.
    """
    def __init__(self, chain, output_key, prefix="[Chain]", debug_mode=False):
        self.chain = chain
        self.output_key = output_key
        self.logger = ChainLogger(prefix=prefix, debug_mode=debug_mode)

    def invoke(self, inputs, config=None, **kwargs):
        start_time = time.monotonic()
        session_info = f"{inputs.get('session_type', 'n/a')}:{inputs.get('session_id', 'n/a')}"
        text = inputs.get("last message from client", "")

        if not text or not text.strip():
            self.logger.log(session_info, "warning", "пустой текст, модель не вызвалась")
            return {**inputs, self.output_key: EMPTY_MESSAGE}

        self.logger.log(session_info, "info", "started")

        try:
            result = self.chain.invoke(inputs, config=config, **kwargs)
            elapsed = time.monotonic() - start_time

            self.logger.log(session_info, "debug", f"input: {inputs}")
            self.logger.log(session_info, "info", f"finished in {elapsed:.2f}s")
            self.logger.log(session_info, "debug", f"output: {self.output_key}={result}")

            return {**inputs, self.output_key: result}

        except Exception as e:
            self.logger.log(session_info, "error", f"Ошибка: {str(e)}", exc=e)
            return {**inputs, self.output_key: EMPTY_MESSAGE}