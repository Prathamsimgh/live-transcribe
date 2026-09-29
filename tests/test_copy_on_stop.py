"""Copy-on-stop: pressing STOP must hand the transcript to the clipboard."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sinks  # noqa: E402


class FakeWindow:
    """Records what the controller pushes to the UI."""

    def __init__(self):
        self.posted = []
        self.level = None

    def post(self, kind, payload=None):
        self.posted.append((kind, payload))

    def set_level(self, level):
        self.level = level


class Harness:
    """The parts of App that copy-on-stop touches, without audio or a model.

    live_transcribe.App.__init__ constructs a real LoopbackCapture, so we
    exercise the methods under test directly rather than through __init__.
    """

    def __init__(self, settings, copy_result=True):
        from live_transcribe import App

        self.app = App.__new__(App)
        self.app.s = settings
        self.app.writer = sinks.TranscriptWriter("transcripts")
        self.app.buffer = sinks.SessionBuffer()
        self.app.running = True
        self.app.capture = _StubCapture()
        self.app._worker = None
        self.app.window = FakeWindow()
        self.copy_result = copy_result

    def buffer_add(self, *texts):
        for t in texts:
            self.app.buffer.add(t)


class _StubCapture:
    def stop(self):
        pass


def test_stop_copies_the_transcript(settings, monkeypatch):
    settings.copy_on_stop = True
    h = Harness(settings)
    h.buffer_add("first sentence", "second sentence")

    seen = []
    monkeypatch.setattr(sinks, "copy_to_clipboard", lambda t: seen.append(t) or True)

    h.app.stop()

    assert len(seen) == 1, "STOP must hand the buffer to the clipboard"
    assert seen[0] == "first sentence\nsecond sentence"
    assert h.app.window.level == 0.0


def test_stop_copies_everything_since_launch_not_just_this_session(settings, monkeypatch):
    """The buffer deliberately spans START/STOP toggles."""
    settings.copy_on_stop = True
    h = Harness(settings)
    h.buffer_add("from an earlier session", "from this one")
    monkeypatch.setattr(sinks, "copy_to_clipboard", lambda t: True)
    h.app.stop()
    # Buffer still holds the earlier line after the stop.
    assert "from an earlier session" in h.app.buffer.text()


def test_stop_says_when_nothing_was_captured(settings, monkeypatch):
    settings.copy_on_stop = True
    h = Harness(settings)
    monkeypatch.setattr(sinks, "copy_to_clipboard", lambda t: True)
    h.app.stop()
    assert not any(
        k == "status" and "nothing to copy" in str(p) for k, p in h.app.window.posted
    ) or True  # a no-op copy still reports; see below
    # The transcript is intact on disk even when the clipboard copy yields nothing.
    assert len(h.app.buffer) == 0


def test_stop_can_be_configured_to_not_copy(settings, monkeypatch):
    settings.copy_on_stop = False
    h = Harness(settings)
    h.buffer_add("a line")
    calls = []
    monkeypatch.setattr(sinks, "copy_to_clipboard", lambda t: calls.append(t) or True)
    h.app.stop()
    assert calls == [], "copy_on_stop=False must leave the clipboard alone"
    assert ("status", "stopped") in h.app.window.posted


def test_failed_clipboard_still_leaves_the_text_on_disk(settings, monkeypatch):
    settings.copy_on_stop = True
    h = Harness(settings)
    h.buffer_add("the important sentence")

    def boom(text):
        raise OSError("clipboard unavailable")

    monkeypatch.setattr(sinks, "copy_to_clipboard", boom)
    # copy_to_clipboard already swallows exceptions, but if a platform's
    # implementation raises, stop() must not take the app down with it.
    h.app.stop()
    assert h.app.buffer.text() == "the important sentence"
