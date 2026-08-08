"""
Rate Limiter Module

Implements token bucket algorithm to prevent hitting API rate limits.
Designed for Groq API (30 requests/minute free tier).

Features:
- Token bucket rate limiting
- Exponential backoff on failures
- Async-compatible
- Decorator for easy use
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter.

    Default: 30 requests per minute = 0.5 requests per second.
    This matches Groq's free tier limits.
    """

    requests_per_minute: int = 30
    max_retries: int = 3
    base_backoff: float = 1.0
    # Starting token count. Defaults to a full bucket (requests_per_minute) — lets a
    # caller burst up to the full budget immediately, which is fine for Groq's use
    # (a handful of agent calls at cycle start). DhanHQ's account-wide cap doesn't
    # tolerate that: multiple background trackers hitting it within the same second
    # at startup triggered DH-904 even though the sustained rate was well under the
    # limit — pass a small initial_tokens (e.g. 1) for limiters guarding a hard,
    # burst-intolerant cap like that.
    initial_tokens: float | None = None

    # Internal state
    _tokens: float = field(default=0.0, repr=False)
    _last_refill: float = field(default=0.0, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _sync_lock: Any = field(default_factory=Lock, repr=False)

    def __post_init__(self):
        self._tokens = (
            float(self.initial_tokens)
            if self.initial_tokens is not None
            else float(self.requests_per_minute)
        )
        self._last_refill = time.monotonic()

    @property
    def tokens_per_second(self) -> float:
        return self.requests_per_minute / 60.0

    def _refill_tokens(self):
        """Refill tokens based on time elapsed."""
        now = time.monotonic()
        elapsed = now - self._last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.tokens_per_second
        self._tokens = min(
            float(self.requests_per_minute),  # Cap at max
            self._tokens + tokens_to_add,
        )
        self._last_refill = now

    async def acquire(self) -> bool:
        """
        Acquire a token for making a request.

        Blocks until a token is available.

        Returns:
            True when token acquired
        """
        async with self._lock:
            self._refill_tokens()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            # Calculate wait time for next token
            tokens_needed = 1.0 - self._tokens
            wait_time = tokens_needed / self.tokens_per_second

            logger.debug(f"Rate limited. Waiting {wait_time:.2f}s for token...")
            await asyncio.sleep(wait_time)

            self._refill_tokens()
            self._tokens -= 1.0
            return True

    def acquire_sync(self) -> bool:
        """Synchronous version of acquire."""
        with self._sync_lock:
            self._refill_tokens()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            tokens_needed = 1.0 - self._tokens
            wait_time = tokens_needed / self.tokens_per_second

            logger.debug(f"Rate limited. Waiting {wait_time:.2f}s for token...")
            time.sleep(wait_time)

            self._refill_tokens()
            self._tokens -= 1.0
            return True

    def get_wait_time(self) -> float:
        """Get estimated wait time until a token is available."""
        self._refill_tokens()

        if self._tokens >= 1.0:
            return 0.0

        tokens_needed = 1.0 - self._tokens
        return tokens_needed / self.tokens_per_second

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        self._refill_tokens()
        return self._tokens


# Global rate limiter instance for Groq API
_groq_limiter: RateLimiter | None = None


def get_groq_limiter() -> RateLimiter:
    """Get or create the global Groq rate limiter."""
    global _groq_limiter
    if _groq_limiter is None:
        _groq_limiter = RateLimiter(
            requests_per_minute=30,  # Groq free tier
            max_retries=3,
            base_backoff=2.0,
        )
    return _groq_limiter


# Per-provider limiters for the non-Groq LLM providers (see src/agents/llm_factory.py).
# Kept separate from _groq_limiter/get_groq_limiter above rather than folding Groq into
# this dict too, so existing tests that reset the Groq singleton directly
# (`src.utils.rate_limiter._groq_limiter = None`) keep working unchanged.
_provider_limiters: dict[str, RateLimiter] = {}

# Conservative free-tier requests-per-minute defaults. These are deliberately cautious
# estimates (easy to raise for a paid tier) -- the cost of guessing too low is a slower
# agent cycle; the cost of guessing too high is a 429 storm.
_PROVIDER_REQUESTS_PER_MINUTE: dict[str, int] = {
    "gemini": 15,
    "deepseek": 60,
}


def get_llm_provider_limiter(provider: str) -> RateLimiter:
    """Get or create the rate limiter for an LLM provider ("groq"/"gemini"/"deepseek").

    "groq" delegates to get_groq_limiter() (the pre-existing singleton) rather than a
    second bucket, so there is exactly one Groq limiter regardless of which name a
    caller uses to reach it.
    """
    if provider == "groq":
        return get_groq_limiter()
    if provider not in _provider_limiters:
        _provider_limiters[provider] = RateLimiter(
            requests_per_minute=_PROVIDER_REQUESTS_PER_MINUTE.get(provider, 30),
            max_retries=3,
            base_backoff=2.0,
        )
    return _provider_limiters[provider]


# Global rate limiter instance for DhanHQ's Data APIs (quote/ohlc/historical/intraday
# charts). DhanHQ documents a single account-wide cap of 5 requests/second across
# these endpoints *combined*, not a separate budget per endpoint (confirmed live: a
# scalping-screener historical-candle scan and a sector-movers quote scan running
# concurrently each looked fine in isolation but together triggered DH-904 Rate_Limit
# errors) — so every Dhan Data API call site must draw from this one shared bucket.
# initial_tokens=1 matters as much as the sustained rate here: with the default full-
# bucket start, several background trackers all starting at once could burst through
# up to a whole minute's budget (240 requests) in under a second, immediately tripping
# DH-904 again — confirmed live, this exact failure recurred after a restart with
# multiple trackers launching together even though the sustained per-second rate was
# fine.
_dhan_data_api_limiter: RateLimiter | None = None


def get_dhan_data_api_limiter() -> RateLimiter:
    """Get or create the global DhanHQ Data API rate limiter."""
    global _dhan_data_api_limiter
    if _dhan_data_api_limiter is None:
        _dhan_data_api_limiter = RateLimiter(
            requests_per_minute=60,
            max_retries=3,
            base_backoff=2.0,
            initial_tokens=1,
        )
    return _dhan_data_api_limiter


_dhan_quote_api_limiter: RateLimiter | None = None


def get_dhan_quote_api_limiter() -> RateLimiter:
    """Get or create the DhanHQ Quote API limiter (one request per second)."""
    global _dhan_quote_api_limiter
    if _dhan_quote_api_limiter is None:
        _dhan_quote_api_limiter = RateLimiter(
            requests_per_minute=50,
            max_retries=3,
            base_backoff=2.0,
            initial_tokens=1,
        )
    return _dhan_quote_api_limiter


def rate_limited(
    limiter: RateLimiter | None = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> Callable[[F], F]:
    """
    Decorator to rate limit a function.

    Args:
        limiter: RateLimiter instance (uses Groq limiter by default)
        max_retries: Maximum retry attempts on rate limit errors
        backoff_factor: Multiplier for exponential backoff

    Example:
        @rate_limited()
        async def call_groq_api():
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            _limiter = limiter or get_groq_limiter()

            for attempt in range(max_retries + 1):
                await _limiter.acquire()

                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()

                    # Check if it's a rate limit error
                    if "rate" in error_str and "limit" in error_str:
                        if attempt < max_retries:
                            wait_time = backoff_factor**attempt
                            logger.warning(
                                f"Rate limit hit. Retrying in {wait_time:.1f}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue

                    # Re-raise non-rate-limit errors or if retries exhausted
                    raise

            # Should not reach here, but just in case
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            _limiter = limiter or get_groq_limiter()

            for attempt in range(max_retries + 1):
                _limiter.acquire_sync()

                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()

                    if "rate" in error_str and "limit" in error_str:
                        if attempt < max_retries:
                            wait_time = backoff_factor**attempt
                            logger.warning(
                                f"Rate limit hit. Retrying in {wait_time:.1f}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(wait_time)
                            continue

                    raise

            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def test_rate_limiter():
    """Test the rate limiter."""
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("[RATE LIMITER] ₹DeltaQuant - Rate Limiter Test")
    print("=" * 60)

    # Create limiter with 10 RPM for faster testing
    limiter = RateLimiter(requests_per_minute=10)

    print(f"\n[CONFIG] Requests per minute: {limiter.requests_per_minute}")
    print(f"[CONFIG] Tokens per second: {limiter.tokens_per_second:.3f}")

    print("\n[TEST] Making 5 rapid requests...")

    for i in range(5):
        start = time.monotonic()
        limiter.acquire_sync()
        elapsed = (time.monotonic() - start) * 1000

        print(
            f"  Request {i + 1}: acquired in {elapsed:.1f}ms | tokens: {limiter.available_tokens:.1f}"
        )

    print("\n[TEST] Tokens depleted, next request will wait...")

    wait_time = limiter.get_wait_time()
    print(f"  Estimated wait: {wait_time:.2f}s")

    start = time.monotonic()
    limiter.acquire_sync()
    elapsed = time.monotonic() - start
    print(f"  Request 6: acquired after {elapsed:.2f}s wait")

    print("\n" + "=" * 60)
    print("[SUCCESS] Rate limiter working!")
    print("=" * 60)


if __name__ == "__main__":
    test_rate_limiter()
