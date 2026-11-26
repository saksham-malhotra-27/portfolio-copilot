"""Shared pytest setup: create tables once before all tests, close the pool after."""

import pytest

from portfolio_copilot.data.db import get_pool, init_db
from portfolio_copilot.data.http import get_http_client

@pytest.fixture(scope="session", autouse=True)
def database():
    init_db()
    yield
    get_pool().close()
    get_http_client().close()
