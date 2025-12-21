from __future__ import annotations

import requests
from typing import Any, Dict, Optional

from .config import settings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.HTTP_TIMEOUT

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise OllamaError(f"Ollama request failed: {e}") from e

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = requests.get(url, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise OllamaError(f"Ollama request failed: {e}") from e

    def reachable(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except OllamaError:
            return False

    def list_models(self) -> Dict[str, Any]:
        return self._get("/api/tags")

    def pull(self, model: str, stream: bool = False) -> Dict[str, Any]:
        # Non-stream mode returns a final JSON only if stream=False
        return self._post("/api/pull", {"name": model, "stream": stream})

    def chat(self, model: str, messages, temperature: float = 0.2, stream: bool = False) -> Dict[str, Any]:
        # Using /api/chat
        payload = {
            "model": model,
            "messages": [m.model_dump() if hasattr(m, "model_dump") else m for m in messages],
            "options": {"temperature": temperature},
            "stream": stream,
        }
        return self._post("/api/chat", payload)

    def embed(self, model: str, input_list) -> Dict[str, Any]:
        # Using /api/embeddings
        payload = {"model": model, "input": input_list}
        return self._post("/api/embeddings", payload)
