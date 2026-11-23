from enum import StrEnum
from typing import TypedDict


class Source(StrEnum):
    MUTUAL_FUND = "mutualfund"
    STOCK = "stock"


class SchemeSearchResult(TypedDict):
    scheme_code: str
    name: str
