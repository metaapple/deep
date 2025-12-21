from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class HealthResponse(BaseModel):
    ok: bool
    ollama_reachable: bool
    message: str

class ModelInfo(BaseModel):
    name: str
    modified_at: Optional[str] = None
    size: Optional[int] = None
    digest: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class PullRequest(BaseModel):
    model: str = Field(..., description="Ollama model name (e.g., llama3.2:3b) or your custom model")
    stream: bool = False


class PullResponse(BaseModel):
    status: str
    raw: Dict[str, Any]


class ChatMessage(BaseModel):
    role: str = Field(..., description="system|user|assistant")
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.2
    stream: bool = False


class ChatResponse(BaseModel):
    model: str
    response: str
    raw: Dict[str, Any]


class EmbedRequest(BaseModel):
    model: Optional[str] = None
    input: List[str]


class EmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]
    raw: Dict[str, Any]
