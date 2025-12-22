import ollama
MODEL = "llama3.2"

async def stream_generate(word: str):
    stream = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=word,
        stream=True,
        keep_alive=-1
    )
    async for part in stream:
        yield part["response"]