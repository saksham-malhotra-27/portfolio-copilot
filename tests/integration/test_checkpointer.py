"""Integration test: conversations survive a restart with the real Postgres checkpointer."""

import uuid

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command

from portfolio_copilot.agent.checkpointer import get_checkpointer
from portfolio_copilot.agent.graph import build_graph
from portfolio_copilot.data.db import get_pool

pytestmark = pytest.mark.integration

LANGGRAPH_TABLES = {
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
}


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_setup_creates_langgraph_tables():
    get_checkpointer()
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    assert LANGGRAPH_TABLES <= {name for (name,) in rows}


def test_conversation_survives_restart():
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    first_run = build_graph(
        FakeToolModel(messages=iter([AIMessage(content="first answer")])),
        get_checkpointer(),
    )
    first_run.invoke({"messages": []}, config)
    first_run.invoke(Command(resume="first question"), config)

    # a new app object, like restarting the program, with the same thread id
    second_run = build_graph(FakeToolModel(messages=iter([])), get_checkpointer())
    state = second_run.get_state(config)

    assert [m.content for m in state.values["messages"]] == [
        "first question",
        "first answer",
    ]
    assert state.next == ("human",)  # paused, waiting for the next message
