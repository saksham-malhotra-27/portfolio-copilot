"""Shared data models for the data layer."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Source(StrEnum):
    MUTUAL_FUND = "mutualfund"
    STOCK = "stock"


class PricePoint(BaseModel):
    """One day's price (or NAV) for an instrument."""

    model_config = ConfigDict(frozen=True)

    price_date: date
    price: float = Field(gt=0)


class SchemeSearchResult(BaseModel):
    """A mutual fund scheme from a name search."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    scheme_code: str = Field(validation_alias="schemeCode")
    name: str = Field(validation_alias="schemeName")

class Cashflow(BaseModel):
    """One investment or withdrawal. Negative = money invested, positive = money received or current value."""

    model_config = ConfigDict(frozen=True)

    flow_date: date
    amount: float


class DrawdownResult(BaseModel):
    """The worst fall from a previous peak."""

    max_drawdown: float = Field(le=0)
    peak_date: date
    peak_price: float
    trough_date: date
    trough_price: float


class RollingReturnsSummary(BaseModel):
    """Distribution of annualized returns over every window of a given length."""

    window_years: float
    window_count: int
    min_return: float
    median_return: float
    max_return: float
    positive_share: float = Field(ge=0, le=1)
