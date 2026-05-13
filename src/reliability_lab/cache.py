from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Shared utilities — use these in both ResponseCache and SharedRedisCache
# ---------------------------------------------------------------------------

PRIVACY_PATTERNS = re.compile(
    r"\b(balance|password|credit.card|ssn|social.security|user.\d+|account.\d+)\b",
    re.IGNORECASE,
)


def _is_uncacheable(query: str) -> bool:
    """Return True if query contains privacy-sensitive keywords."""
    return bool(PRIVACY_PATTERNS.search(query))


def _looks_like_false_hit(query: str, cached_key: str) -> bool:
    """Return True if query and cached key contain different 4-digit numbers (years, IDs)."""
    nums_q = set(re.findall(r"\b\d{4}\b", query))
    nums_c = set(re.findall(r"\b\d{4}\b", cached_key))
    return bool(nums_q and nums_c and nums_q != nums_c)


# ---------------------------------------------------------------------------
# In-memory cache (existing)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CacheEntry:
    key: str
    value: str
    created_at: float
    metadata: dict[str, str]


class ResponseCache:
    """In-memory cache with semantic similarity and false-hit guardrails."""

    def __init__(self, ttl_seconds: int, similarity_threshold: float):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._entries: list[CacheEntry] = []
        self.false_hit_log: list[dict[str, object]] = []

    def get(self, query: str) -> tuple[str | None, float]:
        if _is_uncacheable(query):
            return None, 0.0

        best_value: str | None = None
        best_score = 0.0
        now = time.time()
        self._entries = [e for e in self._entries if now - e.created_at <= self.ttl_seconds]
        for entry in self._entries:
            score = self.similarity(query, entry.key)
            if score > best_score:
                best_score = score
                best_value = entry.value
                best_key = entry.key

        if best_score >= self.similarity_threshold:
            if _looks_like_false_hit(query, best_key):
                self.false_hit_log.append(
                    {"query": query, "cached_key": best_key, "score": round(best_score, 4)}
                )
                return None, best_score
            return best_value, best_score
        return None, best_score

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        if _is_uncacheable(query):
            return
        self._entries.append(CacheEntry(query, value, time.time(), metadata or {}))

    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Improved similarity using character n-gram overlap + exact-match fast path."""
        if a == b:
            return 1.0

        a_lower = a.lower()
        b_lower = b.lower()

        n = 3
        a_grams = Counter(a_lower[i : i + n] for i in range(len(a_lower) - n + 1))
        b_grams = Counter(b_lower[i : i + n] for i in range(len(b_lower) - n + 1))

        if not a_grams or not b_grams:
            return 0.0

        intersection = sum((a_grams & b_grams).values())
        union = sum((a_grams | b_grams).values())
        return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Redis shared cache (new)
# ---------------------------------------------------------------------------


class SharedRedisCache:
    """Redis-backed shared cache for multi-instance deployments."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        similarity_threshold: float,
        prefix: str = "rl:cache:",
    ):
        import redis as redis_lib

        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.prefix = prefix
        self.false_hit_log: list[dict[str, object]] = []
        self._redis: Any = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return bool(self._redis.ping())
        except Exception:
            return False

    def get(self, query: str) -> tuple[str | None, float]:
        """Look up a cached response from Redis."""
        if _is_uncacheable(query):
            return None, 0.0

        best_value: str | None = None
        best_score = 0.0
        best_cached_query: str | None = None

        try:
            for redis_key in self._redis.scan_iter(f"{self.prefix}*"):
                cached_query = self._redis.hget(redis_key, "query")
                if cached_query is None:
                    continue
                cached_response = self._redis.hget(redis_key, "response")
                score = ResponseCache.similarity(query, cached_query)
                if score > best_score:
                    best_score = score
                    best_value = cached_response
                    best_cached_query = cached_query
        except Exception:
            return None, 0.0

        if best_score >= self.similarity_threshold and best_value is not None:
            if _looks_like_false_hit(query, best_cached_query or ""):
                self.false_hit_log.append(
                    {"query": query, "cached_key": best_cached_query, "score": round(best_score, 4)}
                )
                return None, best_score
            return best_value, best_score
        return None, best_score

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        """Store a response in Redis with TTL."""
        if _is_uncacheable(query):
            return
        try:
            key = f"{self.prefix}{self._query_hash(query)}"
            self._redis.hset(key, mapping={"query": query, "response": value})
            self._redis.expire(key, self.ttl_seconds)
        except Exception:
            pass

    def flush(self) -> None:
        """Remove all entries with this cache prefix (for testing)."""
        for key in self._redis.scan_iter(f"{self.prefix}*"):
            self._redis.delete(key)

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            self._redis.close()

    @staticmethod
    def _query_hash(query: str) -> str:
        """Deterministic short hash for a query string."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]
