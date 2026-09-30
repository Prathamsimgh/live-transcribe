"""faster-whisper wrapper: GPU with automatic CPU fallback.

ctranslate2 will not find pip-installed NVIDIA libraries on its own, so the
wheel DLL directories are registered before faster_whisper is imported.
Import order matters here.
"""

from __future__ import annotations

import os
import re
import site
import string
import sys
from pathlib import Path
from typing import List, Optional


def add_cuda_dll_dirs() -> List[str]:
    """Register pip-installed NVIDIA DLL dirs. Must run before ctranslate2.

    Returns the directories added, for diagnostics.
    """
    if not hasattr(os, "add_dll_directory"):  # non-Windows
        return []

    bases = []
    try:
        bases.extend(site.getsitepackages())
    except Exception:
        pass
    bases.append(os.path.join(sys.prefix, "Lib", "site-packages"))

    added = []
    seen = set()
    for base in bases:
        nvidia = Path(base) / "nvidia"
        if not nvidia.is_dir():
            continue
        for sub in nvidia.iterdir():
            for cand in (sub / "bin", sub):
                if not cand.is_dir():
                    continue
                key = str(cand).lower()
                if key in seen:
                    continue
                try:
                    if not any(cand.glob("*.dll")):
                        continue
                    os.add_dll_directory(str(cand))
                    seen.add(key)
                    added.append(str(cand))
                except OSError:
                    continue

    # ctranslate2's native extension resolves cublas/cudnn via the process
    # DLL search path at *call* time, not just load time. add_dll_directory
    # alone is unreliable for this on some ctranslate2 builds, so also put the
    # dirs on PATH and eagerly load each DLL so it lands in the module table.
    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
        _preload_cuda_dlls(added)
    return added


def _preload_cuda_dlls(dirs: List[str]) -> None:
    """Eagerly load cuBLAS/cuDNN so ctranslate2 finds them by name."""
    import ctypes

    names = [
        "cublas64_12.dll", "cublasLt64_12.dll",
        "cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll",
        "cudnn_adv64_9.dll", "cudnn_graph64_9.dll",
        "cudnn_engines_precompiled64_9.dll", "cudnn_heuristic64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
    ]
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p)
                except OSError:
                    pass


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and whitespace, for blocklist matching."""
    cleaned = text.strip().lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", cleaned).strip()


class Transcriber:
    """Transcribes complete utterances. One model call per utterance."""

    def __init__(self, settings, model_factory=None) -> None:
        self.s = settings
        self.device = "unknown"
        self.compute_type = ""
        self.dll_dirs: List[str] = []
        self.load_error: Optional[str] = None
        self._previous: str = ""

        if model_factory is None:
            self.dll_dirs = add_cuda_dll_dirs()
            model_factory = self._default_factory

        wanted = settings.device
        attempts = []
        if wanted in ("auto", "cuda"):
            attempts.append(("cuda", settings.compute_type_gpu))
        if wanted in ("auto", "cpu"):
            attempts.append(("cpu", settings.compute_type_cpu))
        if not attempts:
            attempts = [("cpu", settings.compute_type_cpu)]

        last = None
        for device, compute in attempts:
            try:
                self.model = model_factory(settings.model, device, compute)
                self.device = device
                self.compute_type = compute
                return
            except Exception as exc:
                last = exc
                self.load_error = f"{device}: {exc}"
        raise RuntimeError(f"Could not load model on any device: {last}")

    @staticmethod
    def _default_factory(model: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        return WhisperModel(model, device=device, compute_type=compute_type)

    # ------------------------------------------------------------ inference

    def transcribe(self, utterance) -> Optional[str]:
        """Return the transcript, or None if the guard rejects it."""
        prompt = self._previous[-200:] if self._previous else None

        segments, _info = self.model.transcribe(
            utterance.samples,
            language="en",
            task="transcribe",
            beam_size=self.s.beam_size,
            vad_filter=False,  # already segmented upstream
            condition_on_previous_text=True,
            initial_prompt=prompt,
            temperature=[0.0, 0.2, 0.4],
        )
        segments = list(segments)
        if not segments:
            return None

        text = " ".join(s.text.strip() for s in segments).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            return None

        if self._is_hallucination(segments, text, utterance):
            return None

        self._previous = text
        return text

    def _is_hallucination(self, segments, text: str, utterance) -> bool:
        avg_logprob = sum(getattr(s, "avg_logprob", 0.0) for s in segments) / len(segments)
        if avg_logprob < self.s.max_logprob_reject:
            return True

        no_speech = max(getattr(s, "no_speech_prob", 0.0) for s in segments)
        if no_speech > self.s.no_speech_reject:
            return True

        # Stock phrases are only suppressed when the audio was also mostly
        # silent -- "Thank you." is a legitimate thing for a speaker to say.
        normalized = _normalize(text)
        blocklist = {_normalize(p) for p in self.s.hallucination_phrases}
        if normalized in blocklist and utterance.mean_vad_prob < 0.6:
            return True
        return False

    def reset_context(self) -> None:
        """Forget the previous sentence, e.g. on a new session."""
        self._previous = ""

    def warmup(self) -> None:
        """Run one throwaway decode so the first real utterance is fast.

        The first ctranslate2 call pays kernel/autotune warmup (measured
        ~1.0s vs ~0.45s steady-state on the RTX 3050 for an 8.5s clip).
        Silence is decoded with the hallucination guard bypassed and the
        result discarded, so it cannot poison the prompt context.
        """
        try:
            import numpy as np

            silence = np.zeros(8000, dtype=np.float32)  # 0.5 s at 16 kHz
            segments, _info = self.model.transcribe(
                silence,
                language="en",
                task="transcribe",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=None,
                temperature=[0.0],
            )
            list(segments)  # force the lazy decode to actually run
        except Exception:
            pass  # warmup is best-effort; real errors surface on transcribe()
