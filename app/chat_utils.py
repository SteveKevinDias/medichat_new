from euriai.langchain import create_chat_model

MODEL = "gpt-4.1-nano"
TEMPERATURE = 0.7


def get_chat_model(api_key: str):
    return create_chat_model(
        api_key=api_key,
        model=MODEL,
        temperature=TEMPERATURE
    )


def ask_chat_model(chat_model, prompt: str):
    response = chat_model.invoke(prompt)
    return response.content