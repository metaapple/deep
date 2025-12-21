from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .ollama_client import OllamaClient, OllamaError
from .schemas import (
    HealthResponse, ModelsResponse, ModelInfo,
    PullRequest, PullResponse,
    ChatRequest, ChatResponse,
    EmbedRequest, EmbedResponse,
)

app = FastAPI(title="Ollama + FastAPI (Hugging Face GGUF optional)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OllamaClient()


@app.get("/health", response_model=HealthResponse)
def health():
    ok = True
    reachable = client.reachable()
    msg = "OK" if reachable else "Ollama is not reachable. Start Ollama first."
    return HealthResponse(ok=ok, ollama_reachable=reachable, message=msg)


@app.get("/models", response_model=ModelsResponse)
def models():
    try:
        data = client.list_models()
        models = []
        for m in data.get("models", []):
            models.append(ModelInfo(**m))
        return ModelsResponse(models=models)
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/pull", response_model=PullResponse)
def pull(req: PullRequest):
    """Pull a model from the Ollama registry.
    NOTE: For Hugging Face GGUF, see README (you create a custom model locally).
    """
    try:
        data = client.pull(model=req.model, stream=req.stream)
        return PullResponse(status="ok", raw=data)
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    model = req.model or settings.DEFAULT_CHAT_MODEL
    try:
        data = client.chat(model=model, messages=req.messages, temperature=req.temperature, stream=req.stream)
        # non-stream output includes: {"message":{"role":"assistant","content":"..."}, ...}
        msg = data.get("message", {}) or {}
        content = msg.get("content", "")
        return ChatResponse(model=model, response=content, raw=data)
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    model = req.model or settings.DEFAULT_EMBED_MODEL
    try:
        data = client.embed(model=model, input_list=req.input)
        # output: {"embeddings":[...], ...}
        embs = data.get("embeddings", [])
        return EmbedResponse(model=model, embeddings=embs, raw=data)
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e))
