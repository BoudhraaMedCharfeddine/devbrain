from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "DevBrain"
    environment: str = "local"
    log_level: str = "INFO"

    # PostgreSQL + pgvector connection (psycopg 3 driver)
    database_url: str = "postgresql://devbrain:devbrain@localhost:5432/devbrain"

    # LLM (Gemini API by default in prod, Ollama switch for local demos)
    llm_provider: str = "gemini"  # "gemini" | "ollama"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    # Ollama (used only when llm_provider == "ollama")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b-instruct"


@lru_cache
def get_settings() -> Settings:
    return Settings()
