"""Tools the agent can call: thin wrappers over the data and analysis layers."""

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel

from portfolio_copilot.analysis.returns import (
    DAYS_PER_YEAR,
    absolute_return,
    cagr,
    rolling_returns,
)
from portfolio_copilot.analysis.risk import annualized_volatility, max_drawdown
from portfolio_copilot.data.models import PricePoint
from portfolio_copilot.data.mutual_funds import get_mf_history, search_mf
from portfolio_copilot.data.stocks import get_stock_history

DEFAULT_SEARCH_LIMIT = 5
MIN_POINTS_FOR_REPORT = 3
PERCENT = 100
PERCENT_DECIMALS = 2


class PerformanceReport(BaseModel):
    """What the LLM gets back from a performance tool. Percentages are already x100."""

    symbol: str
    period_start: date
    period_end: date
    start_price: float
    end_price: float
    absolute_return_pct: float
    cagr_pct: float | None  # None when the period is shorter than a year
    annualized_volatility_pct: float
    max_drawdown_pct: float
    drawdown_peak_date: date
    drawdown_trough_date: date


class RollingReturnsReport(BaseModel):
    """What the LLM gets back from a rolling returns tool. Percentages are already x100."""

    symbol: str
    window_years: float
    window_count: int
    worst_pct: float
    median_pct: float
    best_pct: float
    positive_share_pct: float


def _pct(value: float) -> float:
    return round(value * PERCENT, PERCENT_DECIMALS)


def _last_years(prices: Sequence[PricePoint], years: float | None) -> list[PricePoint]:
    """Keep only the last `years` of history, or everything when years is None."""
    if not prices:
        raise ValueError("No price history available")
    if years is None:
        return list(prices)
    cutoff = prices[-1].price_date - timedelta(days=round(years * DAYS_PER_YEAR))
    return [point for point in prices if point.price_date >= cutoff]


def _performance(
    symbol: str, prices: Sequence[PricePoint], years: float | None
) -> dict:
    window = _last_years(prices, years)
    if len(window) < MIN_POINTS_FOR_REPORT:
        raise ValueError(f"Not enough price history for {symbol} in that period")

    try:
        cagr_pct = _pct(cagr(window))
    except ValueError:
        cagr_pct = None  # under a year: annualizing would exaggerate

    drawdown = max_drawdown(window)
    report = PerformanceReport(
        symbol=symbol,
        period_start=window[0].price_date,
        period_end=window[-1].price_date,
        start_price=window[0].price,
        end_price=window[-1].price,
        absolute_return_pct=_pct(absolute_return(window)),
        cagr_pct=cagr_pct,
        annualized_volatility_pct=_pct(annualized_volatility(window)),
        max_drawdown_pct=_pct(drawdown.max_drawdown),
        drawdown_peak_date=drawdown.peak_date,
        drawdown_trough_date=drawdown.trough_date,
    )
    return report.model_dump(mode="json")


def _rolling(symbol: str, prices: Sequence[PricePoint], window_years: float) -> dict:
    summary = rolling_returns(prices, window_years)
    report = RollingReturnsReport(
        symbol=symbol,
        window_years=summary.window_years,
        window_count=summary.window_count,
        worst_pct=_pct(summary.min_return),
        median_pct=_pct(summary.median_return),
        best_pct=_pct(summary.max_return),
        positive_share_pct=_pct(summary.positive_share),
    )
    return report.model_dump(mode="json")


@tool
def search_mutual_funds(
    query: Annotated[
        str, "Fund name or part of it, e.g. 'parag parikh flexi cap direct growth'"
    ],
    limit: Annotated[int, "Maximum number of results"] = DEFAULT_SEARCH_LIMIT,
) -> list[dict]:
    """Search Indian mutual funds by name and get their scheme codes.

    Call this first whenever the user names a fund, because the other mutual fund
    tools need a scheme code. Prefer 'Direct' and 'Growth' plans unless the user
    asks for something else. If several schemes match, pick the closest one and
    tell the user which one you used.
    """
    return [result.model_dump() for result in search_mf(query, limit)]


@tool
def mutual_fund_performance(
    scheme_code: Annotated[str, "Scheme code from search_mutual_funds, e.g. '122639'"],
    years: Annotated[
        float | None,
        "Look back this many years, e.g. 3 or 5. Leave empty for the full history",
    ] = None,
) -> dict:
    """Returns, risk and worst fall for an Indian mutual fund, from its daily NAV history.

    Gives absolute return, CAGR (yearly growth rate, empty if under a year),
    annualized volatility (how bumpy it is) and max drawdown (worst fall from a peak,
    with dates). All percentages are already multiplied by 100. When comparing funds,
    use the same `years` for each so the periods match.
    """
    return _performance(scheme_code, get_mf_history(scheme_code), years)


@tool
def stock_performance(
    ticker: Annotated[
        str,
        "Yahoo Finance ticker. NSE stocks end in .NS (e.g. TCS.NS, RELIANCE.NS); US tickers have no suffix (e.g. VOO)",
    ],
    years: Annotated[
        float | None,
        "Look back this many years, e.g. 3 or 5. Leave empty for about 5 years of history",
    ] = None,
) -> dict:
    """Returns, risk and worst fall for a stock or ETF, from daily prices adjusted for splits and dividends.

    Gives absolute return, CAGR (empty if under a year), annualized volatility and
    max drawdown with dates. All percentages are already multiplied by 100.
    """
    return _performance(ticker, get_stock_history(ticker), years)


@tool
def mutual_fund_rolling_returns(
    scheme_code: Annotated[str, "Scheme code from search_mutual_funds, e.g. '122639'"],
    window_years: Annotated[float, "Holding period in years, e.g. 1, 3 or 5"],
) -> dict:
    """How consistent a mutual fund has been over every holding period of a given length.

    Computes the yearly return of every `window_years` long period in the fund's history
    and returns the worst, median and best, plus the percentage of periods that made
    money. Use it for questions like 'how likely was I to lose money holding this for
    3 years?' or 'how reliable is this fund?'.
    """
    return _rolling(scheme_code, get_mf_history(scheme_code), window_years)


@tool
def stock_rolling_returns(
    ticker: Annotated[str, "Yahoo Finance ticker, e.g. TCS.NS or VOO"],
    window_years: Annotated[
        float,
        "Holding period in years, e.g. 1 or 3. Stock history covers about 5 years",
    ],
) -> dict:
    """How consistent a stock or ETF has been over every holding period of a given length.

    Returns the worst, median and best yearly return across all `window_years` long
    periods, plus the percentage of periods that made money.
    """
    return _rolling(ticker, get_stock_history(ticker), window_years)


TOOLS = [
    search_mutual_funds,
    mutual_fund_performance,
    stock_performance,
    mutual_fund_rolling_returns,
    stock_rolling_returns,
]
