"""Integration tests for stock data: need internet and the running Postgres."""

import pytest

from portfolio_copilot.data import stocks
from portfolio_copilot.data.stocks import get_stock_history

pytestmark = pytest.mark.integration

TCS = "TCS.NS"


def test_history_is_sorted_and_positive():
    history = get_stock_history(TCS)
    dates = [point.price_date for point in history]
    assert len(history) > 1000  # about 5 years of trading days
    assert dates == sorted(dates)
    assert all(point.price > 0 for point in history)


def test_ticker_is_normalized():
    assert get_stock_history(" tcs.ns ") == get_stock_history(TCS)


def test_fresh_cache_skips_api(monkeypatch):
    get_stock_history(TCS)

    def fail(_):
        raise AssertionError("API should not be called when the cache is fresh")

    monkeypatch.setattr(stocks, "_fetch_price_history", fail)
    assert get_stock_history(TCS)


def test_stale_cache_used_when_api_down(monkeypatch):
    cached = get_stock_history(TCS)

    def api_down(_):
        raise ConnectionError("Yahoo down")

    monkeypatch.setattr(stocks, "is_fresh", lambda *_: False)
    monkeypatch.setattr(stocks, "_fetch_price_history", api_down)
    assert get_stock_history(TCS) == cached


def test_unknown_ticker_raises():
    with pytest.raises(ValueError):
        get_stock_history("NOTAREALTICKER123.NS")
