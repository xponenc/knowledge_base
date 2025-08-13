from llm_utils import call_llm


def run_extractor(role_config, question="", history=None, verbose=False):
    history_content = history[-1] if history else "сообщений нет"
    user_prompt = (
        f"{role_config['instructions']}\n\n"
        f"Вопрос клиента: {question}\n\n"
        f"Предыдущий ответ менеджера: {history_content}\n\nОтвет:"
    )
    return call_llm(
        role_config["system_prompt"],
        user_prompt,
        temp=role_config.get("temperature", 0),
        verbose=verbose
    )
