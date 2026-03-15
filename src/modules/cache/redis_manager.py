# src/modules/cache/redis_manager.py

import redis.asyncio as redis
import json
import os
from typing import Any, Optional


class RedisManager:
    def __init__(self):
        self.redis = None

    async def initialize(self):
        """Connect to Redis on startup."""
        try:
            self.redis = redis.from_url(
                f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}",
                decode_responses=True
                # decode_responses=True means Redis returns strings
                # instead of raw bytes. Much easier to work with.
            )

            # ping() checks if Redis is actually reachable
            await self.redis.ping()
            print("✅ Redis connected")

        except Exception as e:
            # We don't crash if Redis is unavailable —
            # the app still works, just slower (no caching)
            print(f"⚠️ Redis not available: {e} — continuing without cache")
            self.redis = None

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        Returns None if key doesn't exist or Redis is unavailable.
        """
        if not self.redis:
            return None

        try:
            value = await self.redis.get(key)
            if value is None:
                return None
            # Data was stored as JSON string, parse it back
            return json.loads(value)
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: int = 300):
        """
        Save a value to cache.

        key   = unique string identifier
        value = anything (list, dict, string)
        ttl   = time to live in seconds (default 5 minutes)
              = after this time Redis auto-deletes it
        """
        if not self.redis:
            return

        try:
            # Store as JSON string — Redis only stores strings
            await self.redis.set(key, json.dumps(value), ex=ttl)
        except Exception:
            pass  # Cache failure should never crash the app

    async def set_permanent(self, key: str, value: Any):
        """Save with no expiry — stays until manually deleted."""
        if not self.redis:
            return
        try:
            await self.redis.set(key, json.dumps(value))
        except Exception:
            pass

    async def delete(self, key: str):
        """Remove a key from cache — useful when data changes."""
        if not self.redis:
            return
        try:
            await self.redis.delete(key)
        except Exception:
            pass

    async def close(self):
        if self.redis:
            await self.redis.close()


redis_manager = RedisManager()