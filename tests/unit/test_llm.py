"""Unit tests for the LLM factory: checks the arguments, never creates a real model."""

import pytest

from portfolio_copilot.agent import factory as llm_factory
from portfolio_copilot.agent.factory import DEFAULT_MODELS, create_llm
from portfolio_copilot.config import LLMProvider, Settings


@pytest.fixture
def captured_init(monkeypatch):
    """Replace init_chat_model and record what it was called with."""
    calls = {}

    def fake_init_chat_model(model, **kwargs):
        calls.update(model=model, **kwargs)
        return "fake-llm"

    monkeypatch.setattr(llm_factory, "init_chat_model", fake_init_chat_model)
    return calls


@pytest.fixture
def use_settings(monkeypatch):
    """Make the factory see these settings instead of the real .env."""

    def install(**overrides) -> None:
        settings = Settings(
            database_url="postgresql://test", _env_file=None, **overrides
        )
        monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)

    return install


def test_uses_default_model_for_provider(use_settings, captured_init):
    use_settings(google_api_key="google-test-key")

    assert create_llm(LLMProvider.GOOGLE) == "fake-llm"
    assert captured_init["model"] == DEFAULT_MODELS[LLMProvider.GOOGLE]
    assert captured_init["model_provider"] == "google_genai"
    assert captured_init["api_key"] == "google-test-key"


def test_explicit_model_overrides_default(use_settings, captured_init):
    use_settings(anthropic_api_key="anthropic-test-key")

    create_llm(LLMProvider.ANTHROPIC, "claude-sonnet-5-5")
    assert captured_init["model"] == "claude-sonnet-5-5"
    assert captured_init["model_provider"] == "anthropic"


def test_missing_api_key_raises(use_settings, captured_init):
    use_settings(google_api_key="google-test-key", anthropic_api_key=None)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        create_llm(LLMProvider.ANTHROPIC)
    assert captured_init == {}  # never tried to build the model


def test_every_provider_has_a_default_model_and_key_setting():
    for provider in LLMProvider:
        assert provider in llm_factory.DEFAULT_MODELS
        assert provider in llm_factory.API_KEY_SETTINGS
