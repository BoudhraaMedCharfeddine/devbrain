from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Wire structlog + stdlib logging to emit one JSON line per event.

    Call once at app startup. Every logger.info()/.warning()/etc then
    produces a single JSON object on stdout, ready to be shipped to
    Loki/Grafana without parsing.
    """
    # 1. stdlib logging: route everything to stdout at the requested level.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level.upper(),
    )

    # Silence noisy third-party loggers that don't emit our JSON format
    for noisy in ("httpx", "google_genai", "google.genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 2. structlog pipeline: enrich -> serialize to JSON -> stdlib handler.
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for the given module name."""
    return structlog.get_logger(name)
