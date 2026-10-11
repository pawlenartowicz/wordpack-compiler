"""One-line progress output: rewritten in place on a terminal, milestone lines otherwise."""

from __future__ import annotations

import shutil
import sys
import time

_MIN_INTERVAL = 0.1  # seconds between in-place rewrites


class Progress:
    def __init__(self, label: str, stream=None):
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        isatty = getattr(self.stream, "isatty", None)
        self.tty = bool(isatty and isatty())
        self.started = time.monotonic()
        self._last_write: float | None = None
        self._width = 0  # length of the line last drawn in place

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def update(self, text: str, milestone: bool = False) -> None:
        if self.tty:
            now = time.monotonic()
            if self._last_write is not None and now - self._last_write < _MIN_INTERVAL:
                return
            self._last_write = now
            self._draw(self.label + text)
        elif milestone:
            self._emit(self.label + text + "\n")

    def done(self, text: str) -> None:
        line = self.label + text
        if self.tty:
            self._draw(line, end="\n")
        else:
            self._emit(line + "\n")

    def _draw(self, line: str, end: str = "") -> None:
        # A line reaching the last column would wrap, and `\r` only returns to the
        # start of the current row, so each rewrite would add a row.
        line = line[: shutil.get_terminal_size().columns - 1]
        # Padding blanks what a longer previous line left to the right.
        self._emit("\r" + line.ljust(self._width) + end)
        self._width = 0 if end else len(line)

    def _emit(self, s: str) -> None:
        self.stream.write(s)
        self.stream.flush()


def format_time(seconds: float) -> str:
    s = int(seconds)
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_size(n: float) -> str:
    """Decimal units, as the download reports them: 1 MB = 1,000,000 bytes."""
    return f"{n / 1e3:,.0f} KB" if n < 1e6 else f"{n / 1e6:,.0f} MB"


def format_count(n: int) -> str:
    return f"{n:,}"
