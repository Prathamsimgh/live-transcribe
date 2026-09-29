"""Output sinks: the daily transcript file and the clipboard.

The file is the durable record; the clipboard is the shareable view of it.
Every committed line is flushed and fsynced, so killing the process never
costs more than the sentence currently being transcribed.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

BOM = "﻿"


class TranscriptWriter:
    """Append-only daily transcript file, safe for Notepad to read."""

    RETRY_DELAYS = (0.05, 0.2, 0.5)

    def __init__(self, directory, now_fn: Callable[[], datetime] = datetime.now) -> None:
        self.dir = Path(directory)
        self.now_fn = now_fn
        self._pending: List[str] = []
        self._session_open = False

    @property
    def path(self) -> Path:
        return self.dir / f"{self.now_fn().strftime('%Y-%m-%d')}.txt"

    # ----------------------------------------------------------- raw append

    def _append(self, text: str) -> None:
        """Append text, buffering in memory if the file is temporarily locked.

        A transcribed sentence is never dropped because of an I/O error.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path
        new_file = not path.exists()
        payload = "".join(self._pending) + text
        for delay in self.RETRY_DELAYS + (None,):
            try:
                with open(path, "a", encoding="utf-8", newline="") as fh:
                    if new_file:
                        fh.write(BOM)
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                self._pending.clear()
                return
            except OSError:
                if delay is None:
                    self._pending = [payload]  # keep for the next attempt
                    return
                time.sleep(delay)

    # -------------------------------------------------------------- session

    def start_session(self) -> None:
        stamp = self.now_fn().strftime("%Y-%m-%d %H:%M:%S")
        self._append(f"=== Session started {stamp} ===\r\n")
        self._session_open = True

    def write_line(self, text: str, when: Optional[datetime] = None) -> str:
        when = when or self.now_fn()
        line = f"[{when.strftime('%H:%M:%S')}] {text}\r\n"
        self._append(line)
        return line

    def note(self, message: str) -> None:
        """Record an out-of-band event, e.g. an audio device change."""
        self._append(f"=== {message} ===\r\n")

    def stop_session(self) -> None:
        if not self._session_open:
            return
        stamp = self.now_fn().strftime("%Y-%m-%d %H:%M:%S")
        self._append(f"=== Session stopped {stamp} ===\r\n")
        self._session_open = False


class SessionBuffer:
    """Every sentence since the app launched, for the clipboard.

    Spans multiple START/STOP cycles deliberately: the user toggles capture
    around the parts they care about and then copies the lot.
    """

    def __init__(self) -> None:
        self._lines: List[str] = []

    def add(self, text: str) -> None:
        self._lines.append(text)

    def text(self) -> str:
        """Prose for pasting: no headers, no timestamps."""
        return "\n".join(self._lines)

    def clear(self) -> None:
        self._lines.clear()

    def __len__(self) -> int:
        return len(self._lines)


def copy_to_clipboard(text: str) -> bool:
    if not text:
        return False
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception:
        return False
