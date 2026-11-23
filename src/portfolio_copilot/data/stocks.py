"""Stock price history from Yahoo Finance (via yfinance), cached in Postgres."""

import math
from datetime import date

import yfinance as yf

from portfolio_copilot.config import get_settings
from portfolio_copilot.data.db import is_fresh, load_prices, save_prices
from portfolio_copilot.data.types import Source

SOURCE = Source.STOCK
HISTORY_PERIOD = "5y"


def _fetch_price_history(ticker: str) -> list[tuple[date, float]]:
    """Daily closing prices for a ticker over HISTORY_PERIOD, adjusted for splits and dividends."""
    history = yf.Ticker(ticker).history(
        period=HISTORY_PERIOD,
        auto_adjust=True,
        timeout=get_settings().http_timeout_seconds,
    )

    if history.empty:
        raise ValueError(f"No price data for ticker {ticker}")

    return [
        (timestamp.date(), float(close))
        for timestamp, close in history["Close"].items()
        if not math.isnan(close) and close > 0
    ]


def get_stock_history(ticker: str) -> list[tuple[date, float]]:
    """Price history for a ticker (e.g. TCS.NS, VOO), oldest first. Refreshes when the cache is stale."""
    ticker = ticker.strip().upper()

    refresh_error = None
    if not is_fresh(SOURCE, ticker):
        try:
            price_history = _fetch_price_history(ticker)
            save_prices(SOURCE, ticker, None, price_history)
        # yfinance raises many unrelated error types
        except Exception as error:
            refresh_error = error

    price_history = load_prices(SOURCE, ticker)
    if not price_history and refresh_error:
        raise refresh_error
    return price_history
