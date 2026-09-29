"""Transcript file format, durability, and clipboard scope."""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sinks  # noqa: E402
from sinks import SessionBuffer, TranscriptWriter  # noqa: E402


class Clock:
    def __init__(self, *stamps):
        self.stamps = list(stamps)
        self.i = 0

    def __call__(self):
        s = self.stamps[min(self.i, len(self.stamps) - 1)]
        self.i += 1
        return s


def at(h, m, s, day=28):
    return datetime(2026, 9, day, h, m, s)


def test_writes_bom_and_crlf_for_notepad(tmp_path):
    w = TranscriptWriter(tmp_path, now_fn=lambda: at(14, 32, 10))
    w.start_session()
    w.write_line("Hello there.")
    raw = (tmp_path / "2026-09-28.txt").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "Notepad needs the BOM"
    assert b"\r\n" in raw
    assert b"\n\n" not in raw.replace(b"\r\n", b"\n")


def test_line_format_has_timestamp(tmp_path):
    w = TranscriptWriter(tmp_path, now_fn=lambda: at(14, 32, 15))
    w.write_line("So the first question is about hash maps.")
    text = (tmp_path / "2026-09-28.txt").read_text(encoding="utf-8-sig")
    assert "[14:32:15] So the first question is about hash maps." in text


def test_same_day_restart_appends_and_does_not_truncate(tmp_path):
    w = TranscriptWriter(tmp_path, now_fn=lambda: at(10, 0, 0))
    w.start_session()
    w.write_line("first")
    w.stop_session()

    w2 = TranscriptWriter(tmp_path, now_fn=lambda: at(11, 0, 0))
    w2.start_session()
    w2.write_line("second")

    text = (tmp_path / "2026-09-28.txt").read_text(encoding="utf-8-sig")
    assert "first" in text and "second" in text
    assert text.count("=== Session started") == 2


def test_stop_session_writes_footer_once(tmp_path):
    w = TranscriptWriter(tmp_path, now_fn=lambda: at(14, 45, 2))
    w.start_session()
    w.stop_session()
    w.stop_session()  # idempotent, e.g. stop() then shutdown()
    text = (tmp_path / "2026-09-28.txt").read_text(encoding="utf-8-sig")
    assert text.count("=== Session stopped") == 1


def test_fsync_called_for_every_line(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sinks.os, "fsync", lambda fd: calls.append(fd))
    w = TranscriptWriter(tmp_path, now_fn=lambda: at(9, 0, 0))
    w.write_line("a")
    w.write_line("b")
    assert len(calls) == 2


def test_locked_file_buffers_instead_of_losing_a_sentence(tmp_path, monkeypatch):
    import builtins

    w = TranscriptWriter(tmp_path, now_fn=lambda: at(9, 0, 0))
    monkeypatch.setattr(w, "RETRY_DELAYS", ())

    # Shadow the builtin with a module global; Python resolves module globals
    # before builtins, so sinks.open() picks this up.
    def deny(*a, **k):
        raise PermissionError("locked by Notepad")

    monkeypatch.setattr(sinks, "open", deny, raising=False)
    w.write_line("must not be lost")
    assert w._pending, "sentence must be retained when the file is locked"
    assert not (tmp_path / "2026-09-28.txt").exists()

    monkeypatch.setattr(sinks, "open", builtins.open, raising=False)
    w.write_line("next")
    text = (tmp_path / "2026-09-28.txt").read_text(encoding="utf-8-sig")
    assert "must not be lost" in text and "next" in text


def test_note_records_device_change(tmp_path):
    w = TranscriptWriter(tmp_path, now_fn=lambda: at(9, 0, 0))
    w.note("audio device changed to Headphones")
    text = (tmp_path / "2026-09-28.txt").read_text(encoding="utf-8-sig")
    assert "=== audio device changed to Headphones ===" in text


def test_path_rolls_over_at_midnight(tmp_path):
    clock = Clock(at(23, 59, 59, day=28), at(0, 0, 1, day=29))
    w = TranscriptWriter(tmp_path, now_fn=clock)
    assert w.path.name == "2026-09-28.txt"
    assert w.path.name == "2026-09-29.txt"


# ------------------------------------------------------------- clipboard

def test_clipboard_spans_start_stop_cycles_without_timestamps():
    buf = SessionBuffer()
    buf.add("first sentence")
    buf.add("second sentence")
    text = buf.text()
    assert text == "first sentence\nsecond sentence"
    assert "===" not in text
    assert "[" not in text


def test_empty_clipboard_copy_is_refused(monkeypatch):
    assert sinks.copy_to_clipboard("") is False
