from datetime import date, timedelta

import pytest

from portfolio_copilot.analysis.returns import (
    absolute_return,
    cagr,
    rolling_returns,
    xirr,
)
from portfolio_copilot.data.models import Cashflow, PricePoint


def _point(day: date, price: float) -> PricePoint:
    return PricePoint(price_date=day, price=price)


def test_absolute_return():
    prices = [_point(date(2020, 1, 1), 100), _point(date(2021, 1, 1), 150)]
    assert absolute_return(prices) == pytest.approx(0.5)


def test_needs_at_least_two_prices():
    with pytest.raises(ValueError):
        absolute_return([_point(date(2020, 1, 1), 100)])


def test_cagr_doubling_in_two_years():
    prices = [_point(date(2020, 1, 1), 100), _point(date(2022, 1, 1), 200)]
    assert cagr(prices) == pytest.approx(0.4142, abs=1e-3)  # about sqrt(2) - 1


def test_cagr_rejects_short_history():
    prices = [_point(date(2020, 1, 1), 100), _point(date(2020, 7, 1), 110)]
    with pytest.raises(ValueError):
        cagr(prices)


def test_rolling_returns_constant_growth():
    start = date(2020, 1, 1)
    prices = [
        _point(start + timedelta(days=day), 100 * 1.10 ** (day / 365.25))
        for day in range(3 * 366)
    ]
    summary = rolling_returns(prices, window_years=1)
    assert summary.window_count > 0
    assert summary.min_return == pytest.approx(0.10)
    assert summary.max_return == pytest.approx(0.10)
    assert summary.positive_share == 1.0


def test_rolling_returns_window_longer_than_history():
    prices = [_point(date(2020, 1, 1), 100), _point(date(2021, 6, 1), 120)]
    with pytest.raises(ValueError):
        rolling_returns(prices, window_years=3)


def test_xirr_single_investment():
    flows = [
        Cashflow(flow_date=date(2024, 1, 1), amount=-10_000),
        Cashflow(flow_date=date(2025, 1, 1), amount=11_000),
    ]
    expected = 1.1 ** (365 / 366) - 1  # 2024 is a leap year: 366 days
    assert xirr(flows) == pytest.approx(expected, abs=1e-6)


def test_xirr_monthly_sip_has_zero_npv():
    flows = [
        Cashflow(flow_date=date(2024, month, 1), amount=-5_000)
        for month in range(1, 13)
    ]
    flows.append(Cashflow(flow_date=date(2025, 1, 1), amount=66_000))

    rate = xirr(flows)
    first = flows[0].flow_date
    npv = sum(
        f.amount / (1 + rate) ** ((f.flow_date - first).days / 365) for f in flows
    )
    assert npv == pytest.approx(0, abs=0.01)
    assert rate > 0  # 66,000 back on 60,000 invested


def test_xirr_rejects_one_sided_cashflows():
    flows = [Cashflow(flow_date=date(2024, 1, 1), amount=-5_000)]
    with pytest.raises(ValueError):
        xirr(flows)
