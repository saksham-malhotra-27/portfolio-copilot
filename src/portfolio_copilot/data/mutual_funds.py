"""Mutual fund search and NAV history from mfapi.in, cached in Postgres."""

from datetime import date, datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from portfolio_copilot.config import get_settings
from portfolio_copilot.data.db import is_fresh, load_prices, save_prices
from portfolio_copilot.data.models import PricePoint, SchemeSearchResult, Source

BASE_URL = "https://api.mfapi.in/mf"
SOURCE = Source.MUTUAL_FUND
API_DATE_FORMAT = "%d-%m-%Y"
DEFAULT_SEARCH_LIMIT = 10


class _SchemeMeta(BaseModel):
    scheme_name: str | None = None


class _NavResponse(BaseModel):
    meta: _SchemeMeta = _SchemeMeta()
    data: list[dict[str, Any]] = []


class _NavEntry(BaseModel):
    nav_date: date = Field(validation_alias="date")
    nav: float = Field(gt=0)

    @field_validator("nav_date", mode="before")
    @classmethod
    def _parse_api_date(cls, value: str) -> date:
        return datetime.strptime(value, API_DATE_FORMAT).date()


def _fetch_nav_history(scheme_code: str) -> tuple[str | None, list[PricePoint]]:
    """Call the API for one scheme. Returns (scheme name, prices) in API order."""
    response = httpx.get(
        f"{BASE_URL}/{scheme_code}",
        timeout=get_settings().http_timeout_seconds,
    )
    response.raise_for_status()
    payload = _NavResponse.model_validate(response.json())

    if not payload.data:
        raise ValueError(f"No NAV data for scheme {scheme_code}")

    prices = []
    for raw_entry in payload.data:
        try:
            entry = _NavEntry.model_validate(raw_entry)
        except ValidationError:
            # skip malformed rows instead of failing the whole fetch
            continue
        prices.append(PricePoint(price_date=entry.nav_date, price=entry.nav))
    return payload.meta.scheme_name, prices


def get_mf_history(scheme_code: str | int) -> list[PricePoint]:
    """NAV history for a scheme, oldest first. Refreshes from the API when the cache is stale."""
    # the API gives ints, the database stores text
    scheme_code = str(scheme_code)

    refresh_error = None
    if not is_fresh(SOURCE, scheme_code):
        try:
            scheme_name, prices = _fetch_nav_history(scheme_code)
            save_prices(SOURCE, scheme_code, scheme_name, prices)
        except (httpx.HTTPError, ValueError) as error:
            # fall back to whatever is already cached
            refresh_error = error
    prices = load_prices(SOURCE, scheme_code)
    if not prices and refresh_error:
        # no cache and the API failed: nothing to return
        raise refresh_error
    return prices


def search_mf(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[SchemeSearchResult]:
    """Find schemes by name."""
    response = httpx.get(
        f"{BASE_URL}/search",
        params={"q": query},
        timeout=get_settings().http_timeout_seconds,
    )
    response.raise_for_status()
    return [SchemeSearchResult.model_validate(hit) for hit in response.json()[:limit]]
