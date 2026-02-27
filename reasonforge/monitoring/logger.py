"""
ReasonForge - Structured Logging

Configure structured JSON logging using structlog.
Falls back to standard logging if structlog is not available.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


def setup_logging(
    neuron_type: str = "validator",
    uid: int = 0,
    debug: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure structured JSON logging.

    Uses structlog if available, otherwise falls back to standard logging.
    """
    level = logging.DEBUG if debug else logging.INFO

    if HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        log = structlog.get_logger()
        log = log.bind(neuron_type=neuron_type, uid=uid)

        # Also configure stdlib logging for libraries
        logging.basicConfig(
            level=level,
            format="%(message)s",
            stream=sys.stdout,
        )

        return log
    else:
        # Fallback: standard logging
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

        # File handler
        handlers: list = [console_handler]
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            handlers.append(file_handler)

        logging.basicConfig(
            level=level,
            handlers=handlers,
        )

        log = logging.getLogger(f"reasonforge.{neuron_type}")
        log.info("Logging configured (neuron=%s, uid=%d)", neuron_type, uid)
        return log
