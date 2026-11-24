"""Stock price history from Yahoo Finance (via yfinance), cached in Postgres."""

import math

import yfinance as yf

from portfolio_copilot.config import get_settings
from portfolio_copilot.data.db import is_fresh, load_prices, save_prices
from portfolio_copilot.data.models import PricePoint, Source

SOURCE = Source.STOCK
HISTORY_PERIOD = "5y"


def _fetch_price_history(ticker: str) -> list[PricePoint]:
    """Daily closing prices for a ticker over HISTORY_PERIOD, adjusted for splits and dividends."""
    history = yf.Ticker(ticker).history(
        period=HISTORY_PERIOD,
        auto_adjust=True,
        timeout=get_settings().http_timeout_seconds,
    )
    if history.empty:
        raise ValueError(f"No price data for ticker {ticker}")

    return [
        PricePoint(price_date=timestamp.date(), price=float(close))
        for timestamp, close in history["Close"].items()
        if not math.isnan(close) and close > 0
    ]


def get_stock_history(ticker: str) -> list[PricePoint]:
    """Price history for a ticker (e.g. TCS.NS, VOO), oldest first. Refreshes when the cache is stale."""
    ticker = ticker.strip().upper()

    refresh_error = None
    if not is_fresh(SOURCE, ticker):
        try:
            prices = _fetch_price_history(ticker)
            save_prices(SOURCE, ticker, None, prices)
        # yfinance raises many unrelated error types
        except Exception as error:
            refresh_error = error

    prices = load_prices(SOURCE, ticker)
    if not prices and refresh_error:
        raise refresh_error
    return prices
