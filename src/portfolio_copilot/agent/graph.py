"""The agent graph: human -> agent -> tools loop, with state saved by the checkpointer."""

from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from portfolio_copilot.agent.tools import TOOLS

SYSTEM_PROMPT = """You are a research assistant for Indian mutual funds and stocks.
Today's date is {today}.

How you work:
- Never recommend buying, selling or holding anything. Explain the numbers and the
  trade offs clearly, and leave the decision to the user.
- Never calculate returns, risk or any other figure yourself. Always get numbers from
  the tools and quote them as returned.
- When the user names a mutual fund, call search_mutual_funds first to get its scheme
  code. Prefer Direct Growth plans unless asked otherwise, and say which scheme you used.
- Stocks use Yahoo Finance tickers: NSE stocks end in .NS (e.g. TCS.NS), US tickers
  have no suffix (e.g. VOO).
- When comparing investments, use the same period (years) for each so the comparison
  is fair, and say which period you used.
- Explain terms like CAGR, volatility, max drawdown and rolling returns in plain
  language the first time they come up.
- If a tool returns an error, tell the user briefly what went wrong and what they
  could try instead.
- Keep answers short and concrete. Lead with the key numbers, then explain them."""

FIRST_QUESTION = "What would you like to research?"


def build_graph(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    """Compile the agent graph for this model, saving every step with the checkpointer."""
    llm_with_tools = llm.bind_tools(TOOLS)

    def human(state: MessagesState) -> dict:
        """Pause and wait for the user, showing the AI's last reply as the prompt."""
        messages = state["messages"]
        question = messages[-1].text if messages else FIRST_QUESTION
        answer = interrupt({"question": question})
        return {"messages": [HumanMessage(content=answer)]}

    def agent(state: MessagesState) -> dict:
        """Ask the LLM what to do next: answer, or call tools."""
        system_message = SystemMessage(
            SYSTEM_PROMPT.format(today=date.today().isoformat())
        )
        response = llm_with_tools.invoke([system_message, *state["messages"]])
        return {"messages": [response]}

    def route_after_agent(state: MessagesState) -> str:
        """Run tools if the LLM asked for any, otherwise hand back to the user."""
        return "tools" if state["messages"][-1].tool_calls else "human"

    graph = StateGraph(MessagesState)
    graph.add_node("human", human)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    graph.add_edge(START, "human")
    graph.add_edge("human", "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "human": "human"}
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
