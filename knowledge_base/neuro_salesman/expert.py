
def create_expert_chain(expert_name: str, expert_params: Dict, debug_mode: bool = False):
    llm = ChatOpenAI(model=expert_params.get("model", DEFAULT_LLM_MODEL), temperature=expert_params.get("temp", 0))

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system}"),
        HumanMessagePromptTemplate.from_template(
            "{instructions}\n\n"
            "Вопрос клиента: {question}\n\n"
            "Хронология предыдущих сообщений диалога: {summary_history}\n\n"
            "Точное саммари: {summary_exact}\n\n"
            "База знаний: {docs_content}\n\n"
            "Ответ: "
        )
    ])

    class ExpertRunnable(Runnable):
        def __init__(self, chain, expert_name, expert_params, debug_mode):
            self.chain = chain
            self.expert_name = expert_name
            self.expert_params = expert_params
            self.debug_mode = debug_mode

        def invoke(self, inputs, config=None, **kwargs):
            if self.expert_name in ["Специалист по Zoom", "Специалист по завершению"]:
                docs_content = ""
            else:
                search_index = self.expert_params.get("search_index")
                base_topic_phrase = inputs.get("topic_phrases", AIMessage(content="")).content
                knowledge_base = search_index.similarity_search(base_topic_phrase, k=self.expert_params.get("k", 5))
                docs_content = re.sub(r'\n{2}', ' ', '\n '.join(
                    [f'\n==================\n' + doc.page_content + '\n' for doc in knowledge_base]
                ))

            if self.debug_mode:
                print(f"\n==================\n")
                print(f"Вопрос клиента: {inputs.get('last_message_from_client')}")
                print(f"Саммари диалога:\n==================\n{inputs.get('summary_history')}")
                print(f"Саммари точное:\n==================\n{inputs.get('summary_exact')}")
                print(f"База знаний:\n==================\n{docs_content}")

            result = self.chain.invoke(
                {
                    "system": self.expert_params.get("system", ""),
                    "instructions": self.expert_params.get("instructions", ""),
                    "question": inputs.get("last_message_from_client", ""),
                    "summary_history": "\n".join(inputs.get("histories", [])),
                    "summary_exact": inputs.get("summary_exact", ""),
                    "docs_content": docs_content
                },
                config=config,
                **kwargs
            )

            answer = result.content
            try:
                answer = answer.split(": ")[1] + " "
            except IndexError:
                answer = answer.lstrip("#3")

            if self.debug_mode:
                print(f"\n==================\n")
                print(f"{result.usage_metadata['total_tokens']} total tokens used (question-answer).")
                print(f"\n==================\n")
                print(f"Ответ {self.expert_name}:\n {answer}")

            return f"{self.expert_name}: {answer}"

    return ExpertRunnable(chain=prompt_template | llm, expert_name=expert_name, expert_params=expert_params, debug_mode=debug_mode)

def process_experts(inputs: Dict) -> Dict:
    """Обрабатывает список специалистов, вызывая их цепочки."""
    router_output = inputs.get("router_output", [])
    if not router_output:
        if inputs.get("debug_mode", False):
            print("[Experts] Ответ диспетчера пуст или некорректен.")
        return {**inputs, "experts_answers": []}

    experts_answers = []
    for key_param in router_output:
        expert_params = EXPERTS.get(key_param, {})
        if not expert_params:
            if inputs.get("debug_mode", False):
                print(f"[Experts] Специалист {key_param} не найден в EXPERTS.")
            continue
        expert_params = expert_params | {
            "question": inputs.get("last_message_from_client", ""),
            "summary_history": "\n".join(inputs.get("histories", [])),
            "summary_exact": inputs.get("summary_exact", ""),
            "base_topic_phrase": inputs.get("topic_phrases", AIMessage(content="")).content,
            "search_index": inputs.get("search_index")
        }
        expert_chain = create_expert_chain(key_param, expert_params, inputs.get("debug_mode", False))
        answer = expert_chain.invoke(inputs)
        experts_answers.append(answer)

    return {**inputs, "experts_answers": experts_answers}