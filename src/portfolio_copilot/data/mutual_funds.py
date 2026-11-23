"""Mutual fund search and NAV history from mfapi.in, cached in Postgres."""

from datetime import date, datetime

import httpx

from portfolio_copilot.config import get_settings
from portfolio_copilot.data.db import is_fresh, load_prices, save_prices
from portfolio_copilot.data.types import SchemeSearchResult, Source

BASE_URL = "https://api.mfapi.in/mf"
SOURCE = Source.MUTUAL_FUND
API_DATE_FORMAT = "%d-%m-%Y"
DEFAULT_SEARCH_LIMIT = 10


def _fetch_nav_history(scheme_code: str) -> tuple[str | None, list[tuple[date, float]]]:
    """Call the API for one scheme. Returns (scheme name, [(date, nav), ...]) in API order."""
    response = httpx.get(
        f"{BASE_URL}/{scheme_code}",
        timeout=get_settings().http_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()

    entries = payload.get("data") or []
    if not entries:
        raise ValueError(f"No NAV data for scheme {scheme_code}")

    scheme_name = payload.get("meta", {}).get("scheme_name")
    nav_history = []
    for entry in entries:
        try:
            nav = float(entry["nav"])
            nav_date = datetime.strptime(entry["date"], API_DATE_FORMAT).date()
        except (KeyError, ValueError):
            continue  # skipping malformed rows instead of failing
        if nav > 0:
            nav_history.append((nav_date, nav))
    return scheme_name, nav_history


def get_mf_history(scheme_code: str | int) -> list[tuple[date, float]]:
    """NAV history for a scheme, oldest first. Refreshes from the API when the cache is stale."""
    # the API gives ints, the database stores text
    scheme_code = str(scheme_code)

    refresh_error = None
    if not is_fresh(SOURCE, scheme_code):
        try:
            scheme_name, nav_history = _fetch_nav_history(scheme_code)
            save_prices(SOURCE, scheme_code, scheme_name, nav_history)
        except (httpx.HTTPError, ValueError) as error:
            # fall back to whatever is already cached
            refresh_error = error

    nav_history = load_prices(SOURCE, scheme_code)
    if not nav_history and refresh_error:
        # no cache and the API failed: nothing to return
        raise refresh_error
    return nav_history


def search_mf(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[SchemeSearchResult]:
    """Find schemes by name"""
    response = httpx.get(
        f"{BASE_URL}/search",
        params={"q": query},
        timeout=get_settings().http_timeout_seconds,
    )
    response.raise_for_status()
    return [
        SchemeSearchResult(
            scheme_code=str(scheme["schemeCode"]), name=scheme["schemeName"]
        )
        for scheme in response.json()[:limit]
    ]
