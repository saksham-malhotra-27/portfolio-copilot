import argparse

from portfolio_copilot.agent.factory import create_llm
from portfolio_copilot.config import LLMProvider, get_settings


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Portfolio research copilot")
    parser.add_argument(
        "--provider",
        type=LLMProvider,
        choices=list(LLMProvider),
        default=settings.llm_provider,
    )
    parser.add_argument(
        "--model",
        default=settings.llm_model,
        help="Defaults to the provider's default model",
    )
    parser.add_argument(
        "--thread",
        default="default",
        help="Conversation id, reuse it to continue a session",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    llm = create_llm(args.provider, args.model)
