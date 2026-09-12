from fastapi import HTTPException, status
from redis.asyncio import Redis


class RateLimiter:
    """
    Redis-backed fixed-window rate limiter.

    The identifier can be:
    - user ID
    - IP address
    - email
    - any other stable identifier
    """

    def __init__(
        self,
        redis: Redis,
        max_requests: int,
        window_seconds: int,
        key_prefix: str = "rate_limit",
    ) -> None:
        self.redis = redis
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    async def check(
        self,
        identifier: str | int,
    ) -> None:
        key = f"{self.key_prefix}:{identifier}"

        count = await self.redis.incr(key)

        # Set expiration only for the first request
        # in the current window.
        if count == 1:
            await self.redis.expire(
                key,
                self.window_seconds,
            )

        if count > self.max_requests:
            ttl = await self.redis.ttl(key)
            retry_after = max(ttl, 0)

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": "Rate limit exceeded.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                },
            )
from fastapi import HTTPException, status
from redis.asyncio import Redis


class RateLimiter:
    """
    Redis-backed fixed-window rate limiter.

    The identifier can be:
    - user ID
    - IP address
    - email
    - any other stable identifier
    """

    def __init__(
        self,
        redis: Redis,
        max_requests: int,
        window_seconds: int,
        key_prefix: str = "rate_limit",
    ) -> None:
        self.redis = redis
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    async def check(
        self,
        identifier: str | int,
    ) -> None:
        key = f"{self.key_prefix}:{identifier}"

        count = await self.redis.incr(key)

        # Set expiration only for the first request
        # in the current window.
        if count == 1:
            await self.redis.expire(
                key,
                self.window_seconds,
            )

        if count > self.max_requests:
            ttl = await self.redis.ttl(key)
            retry_after = max(ttl, 0)

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": "Rate limit exceeded.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                },
            )
