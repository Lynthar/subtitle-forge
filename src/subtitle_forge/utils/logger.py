"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_level: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Default level applied to the root logger and (when not
            overridden) to both the console and file handlers.
        log_file: Optional log file path. The file handler always captures
            DEBUG so the file is useful for post-mortem analysis even when
            console is quieter.
        console_level: Optional override for the console handler. Pass
            "INFO" together with level="DEBUG" to keep DEBUG out of the
            user's terminal (where Rich would render every third-party
            stack trace) while still writing it all to the log file.
    """
    handlers = []

    # Console handler with Rich. By default mirrors `level`; when the caller
    # explicitly passes a higher console_level, the terminal stays quiet
    # even if `level` is DEBUG (used by --save-debug-log to send DEBUG to
    # the file without spamming the user's terminal with torio/torch
    # extension-load fallback tracebacks).
    effective_console_level = console_level if console_level is not None else level
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
    )
    console_handler.setLevel(effective_console_level)
    handlers.append(console_handler)

    # File handler — always DEBUG so saved logs are diagnostic-ready.
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    # Root logger must be at the most permissive level so DEBUG actually
    # reaches the file handler when requested. Per-handler levels above
    # then filter what each destination actually shows.
    root_level = "DEBUG" if log_file else level
    logging.basicConfig(
        level=root_level,
        handlers=handlers,
        force=True,
    )

    # Silence DEBUG/INFO from third-party libraries that are otherwise
    # extremely noisy. These would dump native-extension probing,
    # alignment-model migration messages, and HTTP transport details into
    # the user's run.log without any value for subtitle debugging.
    NOISY_THIRD_PARTY = (
        "faster_whisper",
        "httpx",
        "httpcore",
        "torio",                  # FFmpeg extension probing fallbacks
        "torch",
        "torchaudio",
        "torio._extension",
        "speechbrain",
        "pyannote",
        "pytorch_lightning",
        "huggingface_hub",
        "urllib3",
        "matplotlib",
        "matplotlib.font_manager",
    )
    for name in NOISY_THIRD_PARTY:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
