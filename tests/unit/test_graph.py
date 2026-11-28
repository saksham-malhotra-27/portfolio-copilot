"""Unit tests for the graph's routing, with a fake LLM and an in-memory checkpointer."""

import uuid
from datetime import date, timedelta

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from portfolio_copilot.agent import tools
from portfolio_copilot.agent.graph import FIRST_QUESTION, build_graph
from portfolio_copilot.data.models import PricePoint


class FakeToolModel(GenericFakeChatModel):
    """Returns pre-written replies in order. bind_tools is a no-op so the graph can use it."""

    def bind_tools(self, tools, **kwargs):
        return self


def _fake_llm(*replies: AIMessage) -> FakeToolModel:
    return FakeToolModel(messages=iter(replies))


def _new_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def _growth_series(days: int = 800) -> list[PricePoint]:
    start = date(2020, 1, 1)
    return [
        PricePoint(
            price_date=start + timedelta(days=day), price=100 * 1.1 ** (day / 365.25)
        )
        for day in range(days)
    ]


def test_first_turn_asks_the_first_question():
    app = build_graph(_fake_llm(), InMemorySaver())
    result = app.invoke({"messages": []}, _new_config())

    assert result["__interrupt__"][0].value["question"] == FIRST_QUESTION


def test_plain_answer_goes_back_to_the_user():
    app = build_graph(_fake_llm(AIMessage(content="Hello Saksham")), InMemorySaver())
    config = _new_config()

    app.invoke({"messages": []}, config)
    result = app.invoke(Command(resume="hi"), config)

    assert result["__interrupt__"][0].value["question"] == "Hello Saksham"
    messages = app.get_state(config).values["messages"]
    assert isinstance(messages[0], HumanMessage) and messages[0].content == "hi"
    assert isinstance(messages[1], AIMessage) and messages[1].content == "Hello Saksham"


def test_tool_call_runs_the_tool_then_answers(monkeypatch):
    monkeypatch.setattr(tools, "get_mf_history", lambda _code: _growth_series())
    llm = _fake_llm(
        _tool_call("mutual_fund_performance", {"scheme_code": "122639"}),
        AIMessage(content="The CAGR is about 10%"),
    )
    app = build_graph(llm, InMemorySaver())
    config = _new_config()

    app.invoke({"messages": []}, config)
    result = app.invoke(Command(resume="how did this fund do?"), config)

    assert result["__interrupt__"][0].value["question"] == "The CAGR is about 10%"
    tool_messages = [
        m
        for m in app.get_state(config).values["messages"]
        if isinstance(m, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert "cagr_pct" in tool_messages[0].content


def test_tool_error_is_sent_back_to_the_llm(monkeypatch):
    two_points = _growth_series(days=2)  # too short: the tool raises ValueError
    monkeypatch.setattr(tools, "get_mf_history", lambda _code: two_points)
    llm = _fake_llm(
        _tool_call("mutual_fund_performance", {"scheme_code": "122639"}),
        AIMessage(content="Sorry, not enough data for that fund"),
    )
    app = build_graph(llm, InMemorySaver())
    config = _new_config()

    app.invoke({"messages": []}, config)
    result = app.invoke(Command(resume="how did this fund do?"), config)

    tool_message = next(
        m
        for m in app.get_state(config).values["messages"]
        if isinstance(m, ToolMessage)
    )
    assert tool_message.status == "error"
    assert "Not enough price history" in tool_message.content
    assert (
        result["__interrupt__"][0].value["question"]
        == "Sorry, not enough data for that fund"
    )


def test_threads_are_kept_separate():
    checkpointer = InMemorySaver()
    app = build_graph(_fake_llm(AIMessage(content="Answer for thread A")), checkpointer)
    thread_a, thread_b = _new_config(), _new_config()

    app.invoke({"messages": []}, thread_a)
    app.invoke(Command(resume="question in thread A"), thread_a)

    assert len(app.get_state(thread_a).values["messages"]) == 2
    assert app.get_state(thread_b).values == {}  # thread B has never run
