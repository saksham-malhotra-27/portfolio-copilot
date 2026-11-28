"""Command line entry point: pick a model, then chat with the agent in the terminal."""

import argparse
import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from portfolio_copilot.agent.checkpointer import close_checkpointer, get_checkpointer
from portfolio_copilot.agent.factory import create_llm
from portfolio_copilot.agent.graph import build_graph
from portfolio_copilot.config import LLMProvider, get_settings
from portfolio_copilot.data.db import get_pool, init_db
from portfolio_copilot.data.http import get_http_client

EXIT_COMMANDS = {"exit", "quit"}
RECURSION_LIMIT = 50
AGENT_NODE = "agent"


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
        default=None,
        help="Conversation id to continue. Omit to start a new one",
    )
    return parser.parse_args()


def _get_config(thread_id: str, recursion_limit: int = RECURSION_LIMIT) -> RunnableConfig:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }


def _pending_question(app: CompiledStateGraph, config: RunnableConfig) -> str | None:
    """The question the thread is paused on, or None if it isn't waiting for the user."""
    for task in app.get_state(config).tasks:
        for interrupt in task.interrupts:
            return interrupt.value["question"]
    return None


def _stream_turn(app: CompiledStateGraph, graph_input, config: RunnableConfig) -> str:
    """Run the graph until it pauses, printing AI text and tool calls live. Returns the last AI text."""
    last_text = ""
    current_id = None

    for part in app.stream(graph_input, config, stream_mode=["messages", "updates"], version="v2"):
        if part["type"] == "messages":
            message, metadata = part["data"]
            if metadata.get("langgraph_node") != AGENT_NODE or not message.text:
                continue
            if message.id != current_id:
                if current_id is not None:
                    print()
                current_id, last_text = message.id, ""
                print("AI: ", end="")
            print(message.text, end="", flush=True)
            last_text += message.text

        elif part["type"] == "updates":
            agent_update = part["data"].get(AGENT_NODE)
            if not agent_update:
                continue
            for message in agent_update["messages"]:
                for tool_call in getattr(message, "tool_calls", []):
                    if current_id is not None:      # end the streamed line before the tool note
                        print()
                        current_id = None
                    print(f"  [calling {tool_call['name']}]")

    if current_id is not None:
        print()
    return last_text


def main() -> None:
    args = _parse_args()
    llm = create_llm(args.provider, args.model)

    init_db()
    checkpointer = get_checkpointer()
    app = build_graph(llm, checkpointer)

    thread_id = args.thread or str(uuid.uuid4())
    config = _get_config(thread_id)
    print(f"Thread: {thread_id}\n")

    try:
        streamed = ""
        if _pending_question(app, config) is None:
            # not waiting for the user: start a new thread, or finish an interrupted turn
            graph_input = None if app.get_state(config).next else {"messages": []}
            streamed = _stream_turn(app, graph_input, config)

        while (question := _pending_question(app, config)) is not None:
            if question.strip() != streamed.strip():
                print(f"AI: {question}")

            user_input = input("\nYou: ").strip()
            if user_input.lower() in EXIT_COMMANDS:
                break
            if not user_input:
                # don't print the same question again
                streamed = question
                continue

            print()
            streamed = _stream_turn(app, Command(resume=user_input), config)

    except KeyboardInterrupt:
        print()
    finally:
        print(f"\nConversation saved. Continue with --thread {thread_id}")
        get_pool().close()
        close_checkpointer()
        get_http_client().close()
