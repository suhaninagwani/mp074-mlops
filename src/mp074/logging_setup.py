"""JSON logging: one JSON object per line, written to stdout (never to a file)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Format each record as {"ts", "level", "logger", "message", ...extra}."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def setup_json_logging(level: int = logging.INFO) -> logging.Logger:
    """Send JSON lines to stdout. Calling it twice does not add a second handler."""
    logger = logging.getLogger("mp074")
    logger.setLevel(level)
    if not any(getattr(h, "_mp074_json", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler._mp074_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False
    return logger
