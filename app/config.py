from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "DevBrain"
    environment: str = "local"
    log_level: str = "INFO"

    # Added at their respective steps:
    # database_url: str        # step 2 (pgvector)
    # embedding_model: str     # step 3
    # anthropic_api_key: str   # step 4 (LLM)


@lru_cache
def get_settings() -> Settings:
    return Settings()