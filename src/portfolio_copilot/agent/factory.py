"""Build the chat model for the chosen provider."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from portfolio_copilot.config import LLMProvider, get_settings

DEFAULT_MODELS = {
    LLMProvider.GOOGLE: "gemini-3.5-flash-lite",
    LLMProvider.ANTHROPIC: "claude-haiku-5-5",
}

API_KEY_SETTINGS = {
    LLMProvider.GOOGLE: "google_api_key",
    LLMProvider.ANTHROPIC: "anthropic_api_key",
}


def create_llm(provider: LLMProvider, model: str | None = None) -> BaseChatModel:
    """Chat model for a provider, using it's default model when none is given"""
    key_setting = API_KEY_SETTINGS[provider]
    api_key = getattr(get_settings(), key_setting)
    if api_key is None:
        raise ValueError(f"Set {key_setting.upper()} in .env to use {provider.name}")

    return init_chat_model(
        model or DEFAULT_MODELS[provider],
        model_provider=provider.value,
        api_key=api_key.get_secret_value(),
    )
