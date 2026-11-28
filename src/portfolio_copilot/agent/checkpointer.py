"""LangGraph checkpointer backed by Postgres, with its own connection pool."""

from functools import cache

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from portfolio_copilot.config import get_settings

CONNECT_TIMEOUT_SECONDS = 5


@cache
def _checkpoint_pool() -> ConnectionPool:
    """A separate pool, because LangGraph needs different connection settings than the data layer like autocommit"""
    settings = get_settings()
    return ConnectionPool(
        settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": 0,
            "connect_timeout": CONNECT_TIMEOUT_SECONDS,
        },
        open=True,
    )


@cache
def get_checkpointer() -> PostgresSaver:
    """One checkpointer for the whole app. Creates LangGraph's tables on first use."""
    checkpointer = PostgresSaver(_checkpoint_pool())
    checkpointer.setup()
    return checkpointer


def close_checkpointer() -> None:
    _checkpoint_pool().close()
