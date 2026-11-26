"""Risk metrics: daily returns, annualized volatility and maximum drawdown."""

import math
import statistics
from collections.abc import Sequence
from itertools import pairwise

from portfolio_copilot.data.models import DrawdownResult, PricePoint

TRADING_DAYS_PER_YEAR = 252


def daily_returns(prices: Sequence[PricePoint]) -> list[float]:
    """Return from each price to the next, e.g. [0.01, -0.02, ...]."""
    if len(prices) < 2:
        raise ValueError(f"Need at least 2 prices, got {len(prices)}")
    return [
        current.price / previous.price - 1 for previous, current in pairwise(prices)
    ]


def annualized_volatility(prices: Sequence[PricePoint]) -> float:
    """Standard deviation of daily returns, scaled to a year."""
    returns = daily_returns(prices)
    if len(returns) < 2:
        raise ValueError("Volatility needs at least 3 prices")
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(prices: Sequence[PricePoint]) -> DrawdownResult:
    """The worst fall from a previous peak, with the dates and prices involved."""
    if not prices:
        raise ValueError("Need at least 1 price")

    peak = worst_peak = worst_trough = prices[0]
    worst_drawdown = 0.0
    for point in prices:
        if point.price > peak.price:
            peak = point
            continue
        drawdown = point.price / peak.price - 1
        if drawdown < worst_drawdown:
            worst_drawdown, worst_peak, worst_trough = drawdown, peak, point

    return DrawdownResult(
        max_drawdown=worst_drawdown,
        peak_date=worst_peak.price_date,
        peak_price=worst_peak.price,
        trough_date=worst_trough.price_date,
        trough_price=worst_trough.price,
    )
