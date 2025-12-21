from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # If you run with docker-compose, this becomes http://ollama:11434
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Default models. You can override per-request.
    DEFAULT_CHAT_MODEL: str = "llama3.2:3b"
    DEFAULT_EMBED_MODEL: str = "nomic-embed-text"

    # Timeouts (seconds)
    HTTP_TIMEOUT: int = 120

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
