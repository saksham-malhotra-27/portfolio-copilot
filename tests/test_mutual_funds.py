"""Integration tests for mutual fund data: need internet and the running Postgres."""

import httpx
import pytest

from portfolio_copilot.data import mutual_funds
from portfolio_copilot.data.mutual_funds import get_mf_history, search_mf

pytestmark = pytest.mark.integration

PARAG_PARIKH_FLEXI_CAP_DIRECT = "122639"


def test_search_finds_scheme():
    results = search_mf("parag parikh flexi cap direct growth")
    assert any(
        result.scheme_code == PARAG_PARIKH_FLEXI_CAP_DIRECT for result in results
    )


def test_search_respects_limit():
    assert len(search_mf("hdfc", limit=3)) <= 3


def test_history_is_sorted_and_positive():
    history = get_mf_history(PARAG_PARIKH_FLEXI_CAP_DIRECT)
    dates = [point.price_date for point in history]
    assert len(history) > 1000
    assert dates == sorted(dates)
    assert all(point.price > 0 for point in history)


def test_int_scheme_code_hits_same_cache():
    assert get_mf_history(int(PARAG_PARIKH_FLEXI_CAP_DIRECT)) == get_mf_history(
        PARAG_PARIKH_FLEXI_CAP_DIRECT
    )


def test_fresh_cache_skips_api(monkeypatch):
    get_mf_history(PARAG_PARIKH_FLEXI_CAP_DIRECT)  # make sure it's cached

    def fail(_):
        raise AssertionError("API should not be called when the cache is fresh")

    monkeypatch.setattr(mutual_funds, "_fetch_nav_history", fail)
    assert get_mf_history(PARAG_PARIKH_FLEXI_CAP_DIRECT)


def test_stale_cache_used_when_api_down(monkeypatch):
    cached = get_mf_history(PARAG_PARIKH_FLEXI_CAP_DIRECT)

    def api_down(_):
        raise httpx.ConnectError("API down")

    monkeypatch.setattr(
        mutual_funds, "is_fresh", lambda *_: False
    )  # pretend the cache is stale
    monkeypatch.setattr(mutual_funds, "_fetch_nav_history", api_down)
    assert get_mf_history(PARAG_PARIKH_FLEXI_CAP_DIRECT) == cached


def test_unknown_scheme_raises():
    with pytest.raises((ValueError, httpx.HTTPError)):
        get_mf_history("999999999")
