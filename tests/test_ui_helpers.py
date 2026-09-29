"""Pure helpers that do not require a display."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_transcribe import db_level  # noqa: E402
from ui import to_pynput_hotkey  # noqa: E402


@pytest.mark.parametrize("spec,expected", [
    ("ctrl+alt+c", "<ctrl>+<alt>+c"),
    ("ctrl+alt+r", "<ctrl>+<alt>+r"),
    ("ctrl+shift+alt+t", "<ctrl>+<shift>+<alt>+t"),
    ("CTRL+ALT+C", "<ctrl>+<alt>+c"),
    ("f9", "<f9>"),
])
def test_hotkey_conversion(spec, expected):
    assert to_pynput_hotkey(spec) == expected


def test_silence_reads_as_zero_on_the_meter():
    assert db_level(0.0) == 0.0


def test_full_scale_reads_as_one():
    assert db_level(1.0) == pytest.approx(1.0)


def test_meter_is_monotonic_in_loudness():
    levels = [db_level(r) for r in (0.001, 0.01, 0.1, 0.5, 1.0)]
    assert levels == sorted(levels)


def test_meter_never_leaves_the_unit_range():
    for rms in (0.0, 1e-9, 0.5, 1.0, 4.0):
        assert 0.0 <= db_level(rms) <= 1.0
