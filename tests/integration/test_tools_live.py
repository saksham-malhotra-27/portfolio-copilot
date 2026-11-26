"""End to end check that the tools work with the real APIs and database."""

import pytest

from portfolio_copilot.agent import tools

pytestmark = pytest.mark.integration

PARAG_PARIKH_FLEXI_CAP_DIRECT = "122639"


def test_search_then_performance():
    results = tools.search_mutual_funds.invoke(
        {"query": "parag parikh flexi cap direct growth"}
    )
    assert any(r["scheme_code"] == PARAG_PARIKH_FLEXI_CAP_DIRECT for r in results)

    report = tools.mutual_fund_performance.invoke(
        {"scheme_code": PARAG_PARIKH_FLEXI_CAP_DIRECT, "years": 3}
    )
    assert report["cagr_pct"] is not None
    assert report["annualized_volatility_pct"] > 0
    assert report["max_drawdown_pct"] <= 0


def test_stock_rolling_returns():
    report = tools.stock_rolling_returns.invoke({"ticker": "TCS.NS", "window_years": 1})
    assert report["window_count"] > 0
    assert 0 <= report["positive_share_pct"] <= 100
