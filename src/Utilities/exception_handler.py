import logging
import traceback
from typing import Optional


# ─────────────────────────────────────────────────────────────
# ERROR MESSAGE DETAIL
# ─────────────────────────────────────────────────────────────

def error_message_detail(error: Exception) -> str:
    """
    Return detailed information about the original exception.

    Includes:

        - Full filename
        - Function name
        - Line number
        - Original error message
    """

    traceback_info = traceback.extract_tb(
        error.__traceback__
    )

    if not traceback_info:
        return f"Error: {error}"

    last_trace = traceback_info[-1]

    return (
        f"Error in [{last_trace.filename}] "
        f"function [{last_trace.name}] "
        f"at line [{last_trace.lineno}] "
        f"→ Message: {error}"
    )


# ─────────────────────────────────────────────────────────────
# CUSTOM EXCEPTION
# ─────────────────────────────────────────────────────────────

class CustomException(Exception):
    """
    Application-level exception wrapper.

    Responsibilities:
        - Preserve the original exception
        - Preserve the logger for compatibility
        - Generate detailed error information
        - Maintain the original exception as the cause

    This class DOES NOT log.

    Logging remains the responsibility of the caller.
    """

    def __init__(
        self,
        error: Exception,
        logger: Optional[logging.Logger] = None,
    ) -> None:

        self.original_error = error
        self.logger = logger

        self.detailed_error = error_message_detail(
            error
        )

        super().__init__(
            self.detailed_error
        )

    def __str__(self) -> str:
        return self.detailed_error