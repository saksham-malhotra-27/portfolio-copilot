import math
from datetime import date, timedelta

import pytest

from portfolio_copilot.analysis.risk import (
    annualized_volatility,
    daily_returns,
    max_drawdown,
)
from portfolio_copilot.data.models import PricePoint


def _series(values: list[float], start: date = date(2020, 1, 1)) -> list[PricePoint]:
    return [
        PricePoint(price_date=start + timedelta(days=offset), price=value)
        for offset, value in enumerate(values)
    ]


def test_daily_returns():
    assert daily_returns(_series([100, 110, 99])) == pytest.approx([0.10, -0.10])


def test_max_drawdown_finds_peak_and_trough():
    result = max_drawdown(_series([100, 120, 90, 130]))
    assert result.max_drawdown == pytest.approx(-0.25)
    assert result.peak_price == 120
    assert result.trough_price == 90
    assert result.peak_date < result.trough_date


def test_max_drawdown_of_rising_prices_is_zero():
    assert max_drawdown(_series([100, 110, 120])).max_drawdown == 0


def test_volatility_of_constant_price_is_zero():
    assert annualized_volatility(_series([100, 100, 100, 100])) == 0


def test_volatility_known_value():
    # daily returns of +1%, -1%, +1%, -1%
    prices = _series([100, 101, 99.99, 100.9899, 99.980001])
    expected = (
        0.01 * math.sqrt(4 / 3) * math.sqrt(252)
    )  # sample stdev of ±0.01, annualized
    assert annualized_volatility(prices) == pytest.approx(expected, rel=1e-6)
