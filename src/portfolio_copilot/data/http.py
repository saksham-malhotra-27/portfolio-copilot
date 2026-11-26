"""Shared HTTP client for all outbound API calls."""

from functools import cache

import httpx

from portfolio_copilot.config import get_settings

CONNECT_RETRIES = 2


@cache
def get_http_client() -> httpx.Client:
    """One client for the whole app: reuses connections and retries failed connection attempts."""
    return httpx.Client(
        timeout=get_settings().http_timeout_seconds,
        transport=httpx.HTTPTransport(retries=CONNECT_RETRIES),
    )
