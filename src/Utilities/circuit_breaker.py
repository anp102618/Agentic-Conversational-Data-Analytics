import asyncio
import time
from enum import Enum
from typing import Any, Awaitable, Callable

from src.Utilities.exception_handler import CustomException
from src.Utilities.logger_setup import get_log


# ─────────────────────────────────────────────────────────────
# CIRCUIT STATE
# ─────────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# ─────────────────────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Thread-safe asynchronous circuit breaker.

    Logging is resolved through get_log(), which means the
    active pipeline logger is automatically used when the
    circuit breaker runs inside a pipeline context.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 10,
    ) -> None:

        self.logger = get_log()

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.failure_count = 0
        self.last_failure_time = 0.0

        self.state = CircuitState.CLOSED

        self._lock = asyncio.Lock()
        self._half_open_in_progress = False

    # ─────────────────────────────────────────────────────────
    # CAN EXECUTE
    # ─────────────────────────────────────────────────────────

    async def can_execute(self) -> bool:
        """
        Determine whether the protected operation can execute.
        """

        async with self._lock:

            now = time.time()

            # ─────────────────────────────────────────────────
            # OPEN
            # ─────────────────────────────────────────────────

            if self.state == CircuitState.OPEN:

                if (
                    now - self.last_failure_time
                    < self.recovery_timeout
                ):

                    self.logger.warning(
                        "Circuit OPEN | request rejected"
                    )

                    return False

                self.state = CircuitState.HALF_OPEN
                self._half_open_in_progress = False

                self.logger.info(
                    "Circuit HALF_OPEN | probe allowed"
                )

            # ─────────────────────────────────────────────────
            # HALF OPEN
            # ─────────────────────────────────────────────────

            if self.state == CircuitState.HALF_OPEN:

                if self._half_open_in_progress:

                    self.logger.warning(
                        "HALF_OPEN probe already running"
                    )

                    return False

                self._half_open_in_progress = True

                return True

            # ─────────────────────────────────────────────────
            # CLOSED
            # ─────────────────────────────────────────────────

            return True

    # ─────────────────────────────────────────────────────────
    # SUCCESS
    # ─────────────────────────────────────────────────────────

    async def on_success(self) -> None:
        """
        Reset the circuit after a successful operation.
        """

        async with self._lock:

            previous = self.state

            self.failure_count = 0
            self.last_failure_time = 0.0

            self.state = CircuitState.CLOSED

            self._half_open_in_progress = False

            self.logger.info(
                "Circuit CLOSED | previous_state=%s",
                previous.value,
            )

    # ─────────────────────────────────────────────────────────
    # FAILURE
    # ─────────────────────────────────────────────────────────

    async def on_failure(self) -> None:
        """
        Record one complete operation failure.
        """

        async with self._lock:

            self.failure_count += 1
            self.last_failure_time = time.time()

            self._half_open_in_progress = False

            # ─────────────────────────────────────────────────
            # HALF OPEN FAILURE
            # ─────────────────────────────────────────────────

            if self.state == CircuitState.HALF_OPEN:

                self.state = CircuitState.OPEN

                self.logger.warning(
                    "Circuit HALF_OPEN → OPEN | probe failed"
                )

                return

            # ─────────────────────────────────────────────────
            # FAILURE THRESHOLD
            # ─────────────────────────────────────────────────

            if self.failure_count >= self.failure_threshold:

                self.state = CircuitState.OPEN

                self.logger.warning(
                    "Circuit CLOSED → OPEN | failures=%d",
                    self.failure_count,
                )

            else:

                self.logger.warning(
                    "Circuit failure recorded | "
                    "failures=%d/%d",
                    self.failure_count,
                    self.failure_threshold,
                )


# ─────────────────────────────────────────────────────────────
# RESILIENT EXECUTION
# ─────────────────────────────────────────────────────────────

async def execute_with_resilience(
    func: Callable[[], Awaitable[Any]],
    fallback: Callable[[], Awaitable[Any]],
    cb: CircuitBreaker,
    retries: int = 2,
    timeout: int = 10,
    base_delay: float = 0.3,
) -> Any:
    """
    Execute an operation with:

        - Circuit breaking
        - Retry
        - Timeout
        - Exponential backoff
        - Jitter
        - Fallback

    All logs are written through the logger associated with
    the current execution context.
    """

    logger = get_log()

    # ─────────────────────────────────────────────────────────
    # CIRCUIT CHECK
    # ─────────────────────────────────────────────────────────

    if not await cb.can_execute():

        logger.warning(
            "Circuit unavailable | executing fallback"
        )

        return await fallback()

    # retries=2 means:
    #
    # attempt 1
    # attempt 2
    # attempt 3
    #
    # Therefore total attempts = retries + 1.

    attempts = max(
        1,
        retries + 1,
    )

    # ─────────────────────────────────────────────────────────
    # PRIMARY EXECUTION
    # ─────────────────────────────────────────────────────────

    for attempt in range(attempts):

        try:

            logger.info(
                "Primary attempt %d/%d",
                attempt + 1,
                attempts,
            )

            result = await asyncio.wait_for(
                func(),
                timeout=timeout,
            )

            await cb.on_success()

            logger.info(
                "Primary operation succeeded"
            )

            return result

        except Exception as exc:

            logger.warning(
                "Primary attempt %d/%d failed | error=%s",
                attempt + 1,
                attempts,
                str(exc),
            )

            # ────────────────────────────────────────────────
            # RETRIES EXHAUSTED
            # ────────────────────────────────────────────────

            if attempt == attempts - 1:

                await cb.on_failure()

                logger.warning(
                    "Primary operation exhausted retries | "
                    "fallback"
                )

                # ────────────────────────────────────────────
                # FALLBACK
                # ────────────────────────────────────────────

                try:

                    return await fallback()

                except Exception as fallback_exc:

                    logger.exception(
                        "Fallback failed | error=%s",
                        str(fallback_exc),
                    )

                    raise CustomException(
                        fallback_exc,
                        logger,
                    ) from fallback_exc

            # ────────────────────────────────────────────────
            # EXPONENTIAL BACKOFF
            # ────────────────────────────────────────────────

            delay = base_delay * (
                2 ** attempt
            )

            # 10% jitter
            jitter = delay * 0.1

            await asyncio.sleep(
                delay + jitter
            )

    # This should never be reached.
    raise RuntimeError(
        "Resilience execution reached unexpected state"
    )