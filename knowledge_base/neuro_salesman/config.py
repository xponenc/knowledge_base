from langchain_core.messages import AIMessage

LLM_MAX_RETRIES = 3 # Количество попыток вызова LLM при сбое
LLM_TIMEOUT = 30  # секунд - время таймаут на ответ модели

DEFAULT_LLM_MODEL = "gpt-4.1-nano"

EMPTY_MESSAGE = AIMessage(
        content="",
        response_metadata={
            'token_usage': {
                'completion_tokens': 0,
                'prompt_tokens': 0,
                'total_tokens': 0
            },
            'model_name': 'n/a',
            'finish_reason': 'stop'
        },
        usage_metadata={
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }
    )
