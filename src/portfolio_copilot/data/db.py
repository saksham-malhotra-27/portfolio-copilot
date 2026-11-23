"""Postgres connection, schema setup and price cache helpers."""

from datetime import date
from functools import cache

from psycopg_pool import ConnectionPool

from portfolio_copilot.config import get_settings
from portfolio_copilot.data.types import Source

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS instruments (
        source       TEXT NOT NULL CHECK (source IN ('mutualfund', 'stock')),
        symbol       TEXT NOT NULL,          -- scheme code or ticker
        name         TEXT,
        last_fetched TIMESTAMPTZ,
        PRIMARY KEY (source, symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS price_history (
        source TEXT    NOT NULL CHECK (source IN ('mutualfund', 'stock')),
        symbol TEXT    NOT NULL,
        date   DATE    NOT NULL,
        price  NUMERIC NOT NULL,
        PRIMARY KEY (source, symbol, date)
    )
    """,
]


@cache
def get_pool() -> ConnectionPool:
    """One pool for the whole app, created on first use."""
    settings = get_settings()
    return ConnectionPool(
        settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        open=True,
    )


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with get_pool().connection() as conn:
        for statement in SCHEMA:
            conn.execute(statement)


def is_fresh(source: Source, symbol: str, max_age_hours: int | None = None) -> bool:
    """True if this instrument was fetched within max_age_hours (default from settings)."""
    if max_age_hours is None:
        max_age_hours = get_settings().cache_max_age_hours
    with get_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT last_fetched > now() - make_interval(hours => %s)
            FROM instruments
            WHERE source = %s AND symbol = %s
            """,
            (max_age_hours, source.value, symbol),
        ).fetchone()
    return bool(row and row[0])


def save_prices(
    source: Source, symbol: str, name: str | None, prices: list[tuple[date, float]]
) -> None:
    """Insert price rows (skipping dates already saved) and mark the instrument as fetched."""
    with get_pool().connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO price_history (source, symbol, date, price)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source, symbol, date) DO UPDATE SET price = EXCLUDED.price
                """,
                [
                    (source.value, symbol, price_date, price)
                    for price_date, price in prices
                ],
            )
        conn.execute(
            """
            INSERT INTO instruments (source, symbol, name, last_fetched)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (source, symbol)
            DO UPDATE SET name = EXCLUDED.name, last_fetched = now()
            """,
            (source.value, symbol, name),
        )


def load_prices(source: Source, symbol: str) -> list[tuple[date, float]]:
    """All cached prices for an instrument, oldest first."""
    with get_pool().connection() as conn:
        return conn.execute(
            """
            SELECT date, price::float8
            FROM price_history
            WHERE source = %s AND symbol = %s
            ORDER BY date
            """,
            (source.value, symbol),
        ).fetchall()
