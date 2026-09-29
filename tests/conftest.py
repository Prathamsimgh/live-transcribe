import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio import Frame  # noqa: E402
from config import Settings  # noqa: E402


@pytest.fixture
def settings():
    return Settings()


def frame(rms: float, n: int = 512) -> Frame:
    return Frame(samples=np.full(n, rms, dtype=np.float32), rms=rms)


LOUD = 0.2
QUIET = 0.0


def speech_frames(n: int):
    return [frame(LOUD) for _ in range(n)]


def silent_frames(n: int):
    return [frame(QUIET) for _ in range(n)]
