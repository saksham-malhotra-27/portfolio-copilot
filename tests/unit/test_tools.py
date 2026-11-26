"""Unit tests for the agent tools, with fake price data instead of the API and database."""

from datetime import date, timedelta

import pytest

from portfolio_copilot.agent import tools
from portfolio_copilot.data.models import PricePoint, SchemeSearchResult

START = date(2020, 1, 1)
DAYS_PER_YEAR = 365.25


def _growth_series(
    years: float, annual_rate: float = 0.10, start: date = START
) -> list[PricePoint]:
    """Daily prices growing at a perfectly steady yearly rate."""
    total_days = round(years * DAYS_PER_YEAR)
    return [
        PricePoint(
            price_date=start + timedelta(days=day),
            price=100 * (1 + annual_rate) ** (day / DAYS_PER_YEAR),
        )
        for day in range(total_days + 1)
    ]


def _series(values: list[float], start: date = START) -> list[PricePoint]:
    return [
        PricePoint(price_date=start + timedelta(days=offset), price=value)
        for offset, value in enumerate(values)
    ]


@pytest.fixture
def fake_mf_history(monkeypatch):
    """Call it with a list of prices, and the tools will see those prices for any scheme code."""

    def install(prices: list[PricePoint]) -> None:
        monkeypatch.setattr(tools, "get_mf_history", lambda _scheme_code: prices)

    return install


@pytest.fixture
def fake_stock_history(monkeypatch):
    def install(prices: list[PricePoint]) -> None:
        monkeypatch.setattr(tools, "get_stock_history", lambda _ticker: prices)

    return install


def test_performance_report_for_steady_growth(fake_mf_history):
    fake_mf_history(_growth_series(years=2))
    report = tools.mutual_fund_performance.invoke({"scheme_code": "122639"})

    assert report["symbol"] == "122639"
    assert report["period_start"] == "2020-01-01"  # dates come back as JSON strings
    assert report["cagr_pct"] == pytest.approx(10.0, abs=0.01)
    assert report["absolute_return_pct"] == pytest.approx(21.0, abs=0.1)  # 1.1 ** 2 - 1
    assert report["annualized_volatility_pct"] == pytest.approx(0.0, abs=0.01)
    assert report["max_drawdown_pct"] == 0.0


def test_years_limits_the_period(fake_mf_history):
    fake_mf_history(_growth_series(years=6))
    report = tools.mutual_fund_performance.invoke({"scheme_code": "122639", "years": 1})

    period_days = (
        date.fromisoformat(report["period_end"])
        - date.fromisoformat(report["period_start"])
    ).days
    assert period_days == pytest.approx(365, abs=1)
    assert report["cagr_pct"] == pytest.approx(10.0, abs=0.01)


def test_short_period_has_no_cagr(fake_mf_history):
    fake_mf_history(_growth_series(years=2))
    report = tools.mutual_fund_performance.invoke(
        {"scheme_code": "122639", "years": 0.5}
    )

    assert report["cagr_pct"] is None
    assert report["absolute_return_pct"] > 0


def test_drawdown_is_reported_with_dates(fake_mf_history):
    fake_mf_history(_series([100, 120, 90, 130]))
    report = tools.mutual_fund_performance.invoke({"scheme_code": "122639"})

    assert report["max_drawdown_pct"] == pytest.approx(-25.0)
    assert report["drawdown_peak_date"] == "2020-01-02"
    assert report["drawdown_trough_date"] == "2020-01-03"


def test_not_enough_history_raises(fake_mf_history):
    fake_mf_history(_series([100, 110]))
    with pytest.raises(ValueError):
        tools.mutual_fund_performance.invoke({"scheme_code": "122639"})


def test_stock_performance_uses_ticker(fake_stock_history):
    fake_stock_history(_growth_series(years=2))
    report = tools.stock_performance.invoke({"ticker": "TCS.NS", "years": 1})
    assert report["symbol"] == "TCS.NS"
    assert report["cagr_pct"] == pytest.approx(10.0, abs=0.01)


def test_rolling_returns_report(fake_mf_history):
    fake_mf_history(_growth_series(years=3))
    report = tools.mutual_fund_rolling_returns.invoke(
        {"scheme_code": "122639", "window_years": 1}
    )

    assert report["window_count"] > 0
    assert report["worst_pct"] == pytest.approx(10.0, abs=0.01)
    assert report["median_pct"] == pytest.approx(10.0, abs=0.01)
    assert report["best_pct"] == pytest.approx(10.0, abs=0.01)
    assert report["positive_share_pct"] == 100.0


def test_search_returns_plain_dicts(monkeypatch):
    fake_results = [
        SchemeSearchResult.model_validate(
            {"schemeCode": 122639, "schemeName": "Parag Parikh Flexi Cap"}
        )
    ]
    monkeypatch.setattr(tools, "search_mf", lambda _query, _limit: fake_results)

    results = tools.search_mutual_funds.invoke({"query": "parag parikh"})
    assert results == [{"scheme_code": "122639", "name": "Parag Parikh Flexi Cap"}]


def test_every_tool_has_a_unique_name_and_description():
    names = [t.name for t in tools.TOOLS]
    assert len(names) == len(set(names))
    assert all(t.description for t in tools.TOOLS)
