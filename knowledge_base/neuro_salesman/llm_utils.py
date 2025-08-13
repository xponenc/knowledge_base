from openai import OpenAI

openai = OpenAI()


def call_llm(system_prompt, user_prompt, model="gpt-4.1-nano", temp=0, verbose=False):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    answer = completion.choices[0].message.content.strip()
    if verbose:
        print(f"[LLM] Tokens: {completion.usage.total_tokens}")
        print(f"[LLM] Answer: {answer}")
    return answer
