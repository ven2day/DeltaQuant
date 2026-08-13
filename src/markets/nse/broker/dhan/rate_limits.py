"""Dhan account-wide data/quote API rate-limit ownership."""

from __future__ import annotations

from src.core.utils.rate_limiter import RateLimiter

_dhan_data_api_limiter: RateLimiter | None = None
_dhan_quote_api_limiter: RateLimiter | None = None


def get_dhan_data_api_limiter() -> RateLimiter:
    """Return the shared conservative Dhan data-API limiter."""

    global _dhan_data_api_limiter
    if _dhan_data_api_limiter is None:
        _dhan_data_api_limiter = RateLimiter(
            requests_per_minute=240,
            max_retries=3,
            base_backoff=2.0,
            initial_tokens=1,
        )
    return _dhan_data_api_limiter


def get_dhan_quote_api_limiter() -> RateLimiter:
    """Return the shared conservative Dhan quote-API limiter."""

    global _dhan_quote_api_limiter
    if _dhan_quote_api_limiter is None:
        _dhan_quote_api_limiter = RateLimiter(
            requests_per_minute=50,
            max_retries=3,
            base_backoff=2.0,
            initial_tokens=1,
        )
    return _dhan_quote_api_limiter


__all__ = ["get_dhan_data_api_limiter", "get_dhan_quote_api_limiter"]
