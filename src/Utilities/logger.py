import asyncio
import logging
import sys
import time
import tracemalloc
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Callable


# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

tracemalloc.start()


# Current logger for the active pipeline/request context.
#
# Every component that calls get_log() while this context is
# active will receive the same logger.
current_logger: ContextVar[logging.Logger | None] = ContextVar(
    "current_logger",
    default=None,
)


LOG_FORMAT = (
    "[%(asctime)s] "
    "[%(levelname)s] "
    "[%(filename)s:%(lineno)d] "
    "[%(funcName)s] "
    "%(message)s"
)


# ─────────────────────────────────────────────────────────────
# MEMORY HANDLER
# ─────────────────────────────────────────────────────────────

class MemoryHandler(logging.Handler):

    def __init__(self) -> None:
        super().__init__()
        self.logs: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.logs.append(self.format(record))
        except Exception:
            self.handleError(record)


# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────

from datetime import datetime

def setup_logger(log_id: str) -> logging.Logger:
    logger = logging.getLogger(f"app.{log_id}")

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_handler = logging.FileHandler(
        LOG_DIR / f"{log_id}_{timestamp}.log",
        encoding="utf-8",
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    memory_handler = MemoryHandler()
    memory_handler.setLevel(logging.INFO)
    memory_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(memory_handler)

    logger.memory_handler = memory_handler

    return logger


def get_log(name: str | None = None) -> logging.Logger:
    """
    Return the logger for the current execution context.

    Priority:

        1. Active pipeline logger
        2. Component logger

    This is what allows QueryExpansion, QueryExpander,
    CircuitBreaker, etc. to write to ONE log file.
    """

    logger = current_logger.get()

    if logger is not None:
        return logger

    return setup_logger(name or "default")


# ─────────────────────────────────────────────────────────────
# PIPELINE LOGGER CONTEXT
# ─────────────────────────────────────────────────────────────

def set_pipeline_logger(log_id: str):
    """
    Set the logger for the current pipeline context.

    Example:

        token = set_pipeline_logger("QueryExpander")

    All get_log() calls made inside this context will return:

        app.QueryExpander

    which writes to:

        logs/QueryExpander.log
    """

    logger = setup_logger(log_id)

    return current_logger.set(logger)


def reset_pipeline_logger(token) -> None:
    """
    Restore the previous logger context.

    This is important for async applications so that the
    logger context does not leak into another operation.
    """

    current_logger.reset(token)


# ─────────────────────────────────────────────────────────────
# MEMORY
# ─────────────────────────────────────────────────────────────

def format_memory(value: int) -> str:

    sign = "+" if value >= 0 else "-"
    value = abs(value)

    if value < 1024:
        return f"{sign}{value} B"

    if value < 1024 ** 2:
        return f"{sign}{value / 1024:.2f} KB"

    if value < 1024 ** 3:
        return f"{sign}{value / (1024 ** 2):.2f} MB"

    return f"{sign}{value / (1024 ** 3):.2f} GB"


# ─────────────────────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────────────────────

def track_performance(func: Callable):

    @wraps(func)
    async def async_wrapper(*args, **kwargs):

        logger = get_log()

        start_time = time.perf_counter()
        start_current, start_peak = tracemalloc.get_traced_memory()

        try:
            return await func(*args, **kwargs)

        except Exception:

            logger.exception(
                f"{func.__name__} failed",
                stacklevel=2,
            )

            raise

        finally:

            end_time = time.perf_counter()

            end_current, end_peak = (
                tracemalloc.get_traced_memory()
            )

            logger.info(
                f"{func.__name__} completed | "
                f"Time: {end_time - start_time:.2f}s | "
                f"Memory: "
                f"{format_memory(end_current - start_current)} | "
                f"Peak: "
                f"{format_memory(end_peak - start_peak)}",
                stacklevel=2,
            )

    @wraps(func)
    def sync_wrapper(*args, **kwargs):

        logger = get_log()

        start_time = time.perf_counter()
        start_current, start_peak = tracemalloc.get_traced_memory()

        try:
            return func(*args, **kwargs)

        except Exception:

            logger.exception(
                f"{func.__name__} failed",
                stacklevel=2,
            )

            raise

        finally:

            end_time = time.perf_counter()

            end_current, end_peak = (
                tracemalloc.get_traced_memory()
            )

            logger.info(
                f"{func.__name__} completed | "
                f"Time: {end_time - start_time:.2f}s | "
                f"Memory: "
                f"{format_memory(end_current - start_current)} | "
                f"Peak: "
                f"{format_memory(end_peak - start_peak)}",
                stacklevel=2,
            )

    return (
        async_wrapper
        if asyncio.iscoroutinefunction(func)
        else sync_wrapper
    )