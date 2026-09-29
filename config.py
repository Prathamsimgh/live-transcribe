"""Settings for Live Transcribe.

Every tunable lives here with a normative default from the design spec.
Unknown keys in config.json are ignored so a file written by a future
version cannot break startup.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_HALLUCINATION_PHRASES = [
    "thank you",
    "thanks for watching",
    "subtitles by the amara.org community",
    "please subscribe",
    "",
]


@dataclass
class Settings:
    # ASR
    model: str = "small.en"
    device: str = "auto"  # auto | cuda | cpu
    compute_type_gpu: str = "int8_float16"
    compute_type_cpu: str = "int8"
    beam_size: int = 5

    # Audio
    target_rate: int = 16000
    frame_samples: int = 512  # 32.0 ms at 16 kHz; Silero requires exactly this

    # Segmentation
    vad_threshold: float = 0.5
    energy_gate_multiplier: float = 1.5
    speech_start_frames: int = 3
    speech_end_frames: int = 16
    preroll_frames: int = 8
    trailing_keep_frames: int = 3  # ~96 ms of hangover kept after the last word
    min_utterance_ms: int = 320
    max_utterance_s: float = 10.0
    soft_cut_window_s: float = 1.5

    # Hallucination guard
    max_logprob_reject: float = -1.0
    no_speech_reject: float = 0.6
    hallucination_phrases: list = field(
        default_factory=lambda: list(DEFAULT_HALLUCINATION_PHRASES)
    )

    # Output and control
    transcript_dir: str = "transcripts"
    hotkey_copy: str = "ctrl+alt+c"
    hotkey_toggle: str = "ctrl+alt+r"
    always_on_top: bool = True
    copy_on_stop: bool = True

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_samples / self.target_rate

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        """Load settings, ignoring unknown keys. Missing file yields defaults."""
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
