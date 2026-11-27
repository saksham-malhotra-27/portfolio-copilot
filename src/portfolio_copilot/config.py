from enum import StrEnum
from functools import cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class LLMProvider(StrEnum):
    GOOGLE = 'google_genai'
    ANTHROPIC = 'anthropic'

class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    llm_provider: LLMProvider = LLMProvider.GOOGLE
    # None means the provider's default model
    llm_model: str | None = None
    database_url: str
    db_pool_min_size: int = 1
    db_pool_max_size: int = 5
    cache_max_age_hours: int = 12
    http_timeout_seconds: float = 15.0


@cache
def get_settings() -> Settings:
    return Settings()
