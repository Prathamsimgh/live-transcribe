"""Silero VAD v5 via onnxruntime.

Deliberately avoids the `silero-vad` Python wrapper, which imports torch.
We only need the bundled .onnx asset, so this runs the model directly on
onnxruntime and keeps torch out of the environment entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np


def find_model() -> Optional[Path]:
    """Locate silero_vad.onnx, wherever pip put it."""
    candidates = []
    for base in sys.path:
        if not base:
            continue
        p = Path(base)
        candidates.append(p / "silero_vad" / "data" / "silero_vad.onnx")
        candidates.append(p / "silero_vad" / "data" / "silero_vad_16k_op15.onnx")
    local = Path(__file__).parent / "silero_vad.onnx"
    candidates.insert(0, local)
    for c in candidates:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


class SileroVAD:
    """Stateful speech-probability estimator. Call with 512-sample frames.

    Mirrors silero_vad.utils_vad.OnnxWrapper: each 512-sample frame is
    prepended with a 64-sample context window (the tail of the previous
    frame), and the LSTM state is carried between calls. Without the context
    the model's receptive field is misaligned and it reports near-zero
    probability for genuine speech.
    """

    def __init__(self, model_path: Path, rate: int = 16000) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.rate = rate
        self.context_size = 64 if rate == 16000 else 32

        names = {i.name for i in self.sess.get_inputs()}
        self._audio_key = "input" if "input" in names else next(iter(names))
        self._has_state = "state" in names
        self._has_hc = "h" in names and "c" in names
        self._needs_sr = "sr" in names
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.context_size, dtype=np.float32)

    def __call__(self, samples: np.ndarray) -> float:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size != 512:
            x = np.pad(x, (0, max(0, 512 - x.size)))[:512]
        # Prepend context from the previous frame's tail.
        x = np.concatenate([self._context, x]).reshape(1, -1).astype(np.float32)
        self._context = x[0, -self.context_size:].copy()

        feeds = {self._audio_key: x}
        if self._needs_sr:
            feeds["sr"] = np.array(self.rate, dtype=np.int64)
        if self._has_state:
            feeds["state"] = self._state

        out = self.sess.run(None, feeds)
        prob = float(np.asarray(out[0]).reshape(-1)[0])
        if self._has_state and len(out) > 1:
            self._state = out[1]
        return prob


class AlwaysSpeech:
    """Fallback when the Silero asset is missing.

    Everything surviving the segmenter's energy gate counts as speech. This
    degrades gracefully -- music and sound effects get transcribed too --
    rather than failing to start.
    """

    def reset(self) -> None:
        pass

    def __call__(self, samples: np.ndarray) -> float:
        return 1.0


def load_vad(rate: int = 16000):
    """Return a VAD callable and a human-readable description of it."""
    path = find_model()
    if path is None:
        return AlwaysSpeech(), "energy-only (silero_vad.onnx not found)"
    try:
        return SileroVAD(path, rate), f"silero v5 ({path.name})"
    except Exception as exc:  # corrupt asset, bad onnxruntime build
        return AlwaysSpeech(), f"energy-only (silero failed: {exc})"
