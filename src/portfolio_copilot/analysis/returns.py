"""Return metrics: absolute return, CAGR, rolling returns and XIRR."""

import statistics
from collections.abc import Sequence
from datetime import date, timedelta

from portfolio_copilot.data.models import Cashflow, PricePoint, RollingReturnsSummary

# average year length including leap year
DAYS_PER_YEAR = 365.25
# excel's XIRR convention
XIRR_DAYS_PER_YEAR = 365.0
MIN_CAGR_DAYS = 365
MIN_ROLLING_WINDOW_YEARS = 1.0
XIRR_LOWER_RATE = -0.99
XIRR_UPPER_RATE = 10.0
XIRR_RATE_TOLERANCE = 1e-9
XIRR_MAX_ITERATIONS = 200


def _require_points(prices: Sequence[PricePoint], minimum: int = 2) -> None:
    if len(prices) < minimum:
        raise ValueError(f"Need at least {minimum} prices, got {len(prices)}")


def _annualize(start: PricePoint, end: PricePoint) -> float:
    years = (end.price_date - start.price_date).days / DAYS_PER_YEAR
    return (end.price / start.price) ** (1 / years) - 1


def _npv(rate: float, cashflows: Sequence[Cashflow], first_date: date) -> float:
    return sum(
        flow.amount / (1 + rate) ** ((flow.flow_date - first_date).days / XIRR_DAYS_PER_YEAR)
        for flow in cashflows
    )


def absolute_return(prices: Sequence[PricePoint]) -> float:
    """Total return from the first to the last price, e.g. 0.5 for +50%."""
    _require_points(prices)
    return prices[-1].price / prices[0].price - 1


def cagr(prices: Sequence[PricePoint]) -> float:
    """Compound annual growth rate from the first to the last price.

    Raises ValueError for less than a year of history, because annualizing
    short periods exaggerates returns.
    """
    _require_points(prices)
    days = (prices[-1].price_date - prices[0].price_date).days
    if days < MIN_CAGR_DAYS:
        raise ValueError(f"CAGR needs at least {MIN_CAGR_DAYS} days of history, got {days}")
    return _annualize(prices[0], prices[-1])


def rolling_returns(prices: Sequence[PricePoint], window_years: float) -> RollingReturnsSummary:
    """Annualized return of every window_years-long period in the history."""
    _require_points(prices)
    if window_years < MIN_ROLLING_WINDOW_YEARS:
        raise ValueError(f"Window must be at least {MIN_ROLLING_WINDOW_YEARS} year")

    window = timedelta(days=round(window_years * DAYS_PER_YEAR))
    window_returns = []
    end_index = 0
    for start in prices:
        target_date = start.price_date + window
        while end_index < len(prices) and prices[end_index].price_date < target_date:
            end_index += 1
        if end_index == len(prices):
            # no full window left from here on
            break
        window_returns.append(_annualize(start, prices[end_index]))

    if not window_returns:
        raise ValueError(f"History is shorter than the {window_years}-year window")

    return RollingReturnsSummary(
        window_years=window_years,
        window_count=len(window_returns),
        min_return=min(window_returns),
        median_return=statistics.median(window_returns),
        max_return=max(window_returns),
        positive_share=sum(r > 0 for r in window_returns) / len(window_returns),
    )


def xirr(cashflows: Sequence[Cashflow]) -> float:
    """Annualized return for irregular cashflows (e.g. SIPs), matching Excel's XIRR."""
    if not any(flow.amount < 0 for flow in cashflows) or not any(flow.amount > 0 for flow in cashflows):
        raise ValueError("XIRR needs at least one negative (invested) and one positive (value) cashflow")

    first_date = min(flow.flow_date for flow in cashflows)
    low, high = XIRR_LOWER_RATE, XIRR_UPPER_RATE
    npv_low = _npv(low, cashflows, first_date)
    if npv_low * _npv(high, cashflows, first_date) > 0:
        raise ValueError(f"XIRR has no solution between {low:.0%} and {high:.0%}")

    for _ in range(XIRR_MAX_ITERATIONS):
        mid = (low + high) / 2
        npv_mid = _npv(mid, cashflows, first_date)
        if npv_mid == 0 or (high - low) / 2 < XIRR_RATE_TOLERANCE:
            return mid
        # same sign as the low end: the root is above mid
        if npv_mid * npv_low > 0:
            low, npv_low = mid, npv_mid
        else:
            high = mid

    raise ValueError("XIRR did not converge")
