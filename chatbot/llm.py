from langchain_ollama import ChatOllama


def get_llm():
    """Return the local Ollama Llama 3 model."""

    return ChatOllama(
        model="llama3:8b",
        temperature=0.2,
        base_url="http://localhost:11434",
    )


def stream_response(message: str):
    """Stream an LLM response token by token."""

    llm = get_llm()

    for chunk in llm.stream(message):
        if chunk.content:
            yield chunk.content
