"""ASR wrapper: device fallback, prompt threading, hallucination guard.

Uses an injected model factory throughout, so none of this needs a GPU,
faster-whisper, or a downloaded model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr import Transcriber, _normalize  # noqa: E402
from config import Settings  # noqa: E402
from segmenter import Utterance  # noqa: E402


class Seg:
    def __init__(self, text, avg_logprob=-0.2, no_speech_prob=0.05):
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class FakeModel:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(kwargs)
        return iter(self.segments), {}


def utt(mean_vad_prob=0.9, seconds=1.0):
    n = int(16000 * seconds)
    return Utterance(samples=np.zeros(n, dtype=np.float32),
                     mean_vad_prob=mean_vad_prob, duration_s=seconds)


def factory_for(model, fail_on=()):
    def make(name, device, compute_type):
        if device in fail_on:
            raise RuntimeError(f"no {device}")
        model.device, model.compute_type = device, compute_type
        return model
    return make


# --------------------------------------------------------------- fallback

def test_uses_gpu_when_available():
    m = FakeModel([Seg("hello")])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    assert t.device == "cuda"
    assert t.compute_type == "int8_float16"


def test_falls_back_to_cpu_when_cuda_load_fails():
    m = FakeModel([Seg("hello")])
    t = Transcriber(Settings(), model_factory=factory_for(m, fail_on=("cuda",)))
    assert t.device == "cpu"
    assert t.compute_type == "int8"
    assert "cuda" in t.load_error


def test_raises_only_when_every_device_fails():
    m = FakeModel([])
    with pytest.raises(RuntimeError):
        Transcriber(Settings(), model_factory=factory_for(m, fail_on=("cuda", "cpu")))


def test_device_cpu_setting_never_tries_cuda():
    m = FakeModel([Seg("hi")])
    s = Settings(device="cpu")
    t = Transcriber(s, model_factory=factory_for(m, fail_on=("cuda",)))
    assert t.device == "cpu"


# ---------------------------------------------------------------- params

def test_vad_filter_is_disabled_because_we_already_segmented():
    m = FakeModel([Seg("hello")])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    t.transcribe(utt())
    assert m.calls[0]["vad_filter"] is False


def test_previous_sentence_is_threaded_as_initial_prompt():
    m = FakeModel([Seg("first sentence")])
    t = Transcriber(Settings(), model_factory=factory_for(m))

    t.transcribe(utt())
    assert m.calls[0]["initial_prompt"] is None

    m.segments = [Seg("second sentence")]
    t.transcribe(utt())
    assert m.calls[1]["initial_prompt"] == "first sentence"


def test_context_resets_on_demand():
    m = FakeModel([Seg("first")])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    t.transcribe(utt())
    t.reset_context()
    t.transcribe(utt())
    assert m.calls[1]["initial_prompt"] is None


# ----------------------------------------------------- hallucination guard

def test_rejects_low_average_logprob():
    m = FakeModel([Seg("garbled nonsense", avg_logprob=-1.5)])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    assert t.transcribe(utt()) is None


def test_rejects_high_no_speech_probability():
    m = FakeModel([Seg("something", no_speech_prob=0.9)])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    assert t.transcribe(utt()) is None


def test_rejects_stock_phrase_only_when_audio_was_mostly_silent():
    m = FakeModel([Seg("Thank you.")])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    assert t.transcribe(utt(mean_vad_prob=0.2)) is None


def test_keeps_stock_phrase_when_someone_actually_said_it():
    m = FakeModel([Seg("Thank you.")])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    assert t.transcribe(utt(mean_vad_prob=0.95)) == "Thank you."


def test_rejected_text_does_not_poison_the_next_prompt():
    m = FakeModel([Seg("Thank you.")])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    t.transcribe(utt(mean_vad_prob=0.2))
    m.segments = [Seg("real sentence")]
    t.transcribe(utt())
    assert m.calls[1]["initial_prompt"] is None


def test_empty_result_returns_none():
    m = FakeModel([])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    assert t.transcribe(utt()) is None


def test_multiple_segments_are_joined():
    m = FakeModel([Seg(" Hello there."), Seg(" How are you?")])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    assert t.transcribe(utt()) == "Hello there. How are you?"


def test_prompt_is_capped_at_200_chars():
    long_text = "word " * 100
    m = FakeModel([Seg(long_text)])
    t = Transcriber(Settings(), model_factory=factory_for(m))
    t.transcribe(utt())
    t.transcribe(utt())
    assert len(m.calls[1]["initial_prompt"]) <= 200


def test_normalize_strips_punctuation_and_case():
    assert _normalize("  Thank You!! ") == "thank you"
