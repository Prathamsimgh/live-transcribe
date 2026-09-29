"""Settings loading: defaults, unknown keys, malformed files."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402


def test_defaults_match_the_spec():
    s = Settings()
    assert s.model == "small.en"
    assert s.device == "auto"
    assert s.compute_type_gpu == "int8_float16"
    assert s.beam_size == 5
    assert s.target_rate == 16000
    assert s.frame_samples == 512
    assert s.vad_threshold == 0.5
    assert s.speech_start_frames == 3
    assert s.speech_end_frames == 16
    assert s.preroll_frames == 8
    assert s.min_utterance_ms == 320
    assert s.max_utterance_s == 10.0
    assert s.soft_cut_window_s == 1.5


def test_frame_is_32ms_which_silero_requires():
    assert Settings().frame_ms == 32.0


def test_missing_file_yields_defaults(tmp_path):
    s = Settings.load(tmp_path / "nope.json")
    assert s.model == "small.en"


def test_unknown_keys_are_ignored_not_fatal(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"model": "medium.en", "future_option": 42}))
    s = Settings.load(p)
    assert s.model == "medium.en"
    assert not hasattr(s, "future_option")


def test_malformed_json_falls_back_to_defaults(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{ this is not json")
    assert Settings.load(p).model == "small.en"


def test_round_trip_save_and_load(tmp_path):
    p = tmp_path / "config.json"
    original = Settings(model="tiny.en", beam_size=1, always_on_top=False)
    original.save(p)
    loaded = Settings.load(p)
    assert loaded.model == "tiny.en"
    assert loaded.beam_size == 1
    assert loaded.always_on_top is False


def test_hallucination_phrases_are_not_shared_between_instances():
    a, b = Settings(), Settings()
    a.hallucination_phrases.append("mutated")
    assert "mutated" not in b.hallucination_phrases
