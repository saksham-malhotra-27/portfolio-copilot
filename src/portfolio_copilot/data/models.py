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
