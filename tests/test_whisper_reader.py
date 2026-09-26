from __future__ import annotations

import array
import logging
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from voice_core.whisper_reader import WhisperReader, load_faster_whisper

QUIET_PCM = array.array("h", [0, 16384, -16384, 0]).tobytes()


class _Model:
    def __init__(self, *texts):
        self._texts = texts
        self.asked = []

    def transcribe(self, audio, **options):
        self.asked.append((list(audio), options))
        return (SimpleNamespace(text=text) for text in self._texts), None


def test_an_utterance_is_read_as_one_line_with_the_hint_as_whispers_prompt():
    model = _Model(" left", "next. ")

    reading = WhisperReader(load=lambda: model)(QUIET_PCM, "left next, next")

    assert reading == "left next."
    [(samples, options)] = model.asked
    assert samples == pytest.approx([0.0, 0.5, -0.5, 0.0])
    assert (options["initial_prompt"], options["language"]) == ("left next, next", "en")


def test_the_model_is_loaded_once_and_can_be_loaded_ahead_of_the_first_utterance():
    loads = []
    reader = WhisperReader(load=lambda: loads.append(1) or _Model("next"))

    reader.preload()
    reader(QUIET_PCM, "next")
    reader(QUIET_PCM, "next")

    assert loads == [1]


def test_a_load_started_ahead_is_waited_for_by_the_first_reading_rather_than_started_again():
    loading, finish, loads = threading.Event(), threading.Event(), []

    def load():
        loads.append(1)
        loading.set()
        finish.wait(5)
        return _Model("next")

    reader = WhisperReader(load=load)
    reader.load_ahead()
    assert loading.wait(5)
    finish.set()

    assert reader(QUIET_PCM, "next") == "next"
    assert loads == [1]


def test_a_load_ahead_that_failed_is_tried_again_by_the_first_reading(caplog):
    failing, fail, attempts = threading.Event(), threading.Event(), []

    def load():
        attempts.append("tried")
        if not failing.is_set():
            failing.set()
            fail.wait(5)
            raise OSError("the disk was busy")
        return _Model("next")

    reader = WhisperReader(load=load)
    with caplog.at_level(logging.WARNING, logger="voice_core.whisper_reader"):
        reader.load_ahead()
        assert failing.wait(5)
        fail.set()

        assert reader(QUIET_PCM, "next") == "next"

    assert attempts == ["tried", "tried"]
    assert "the disk was busy" in caplog.text


def test_how_long_whisper_took_to_load_is_logged(caplog):
    ticks = iter([10.0, 12.5])

    with caplog.at_level(logging.INFO, logger="voice_core.whisper_reader"):
        WhisperReader(load=_Model, clock=lambda: next(ticks)).preload()

    assert "whisper loaded in 2.5s" in caplog.text


def test_with_no_hint_whisper_is_given_no_prompt_at_all():
    model = _Model("anything at all")

    WhisperReader(load=lambda: model)(QUIET_PCM, "")

    assert model.asked[0][1]["initial_prompt"] is None


def test_a_reader_for_dictation_lifts_a_quiet_microphone_and_lets_whisper_skip_what_is_not_speech():
    model = _Model("make the sky darker")

    WhisperReader(load=lambda: model, for_dictation=True)(QUIET_PCM, "")

    [(samples, options)] = model.asked
    assert samples == pytest.approx([0.0, 0.95, -0.95, 0.0])
    assert options["vad_filter"] is True


def test_a_closer_reading_lifts_a_quiet_microphone_and_lets_whisper_call_nothing_silence():
    model = _Model("next")

    WhisperReader(load=lambda: model).read_closely(QUIET_PCM, "next")

    [(samples, options)] = model.asked
    assert samples == pytest.approx([0.0, 0.95, -0.95, 0.0])
    assert (options["no_speech_threshold"], options["vad_filter"]) == (None, False)


def test_a_reader_for_commands_leaves_the_audio_as_it_was_recorded():
    model = _Model("next")

    WhisperReader(load=lambda: model)(QUIET_PCM, "next")

    assert model.asked[0][1]["vad_filter"] is False


def test_lifting_silence_does_not_divide_by_nothing():
    model = _Model("")

    WhisperReader(load=lambda: model, for_dictation=True)(bytes(8), "")

    assert model.asked[0][0] == [0.0, 0.0, 0.0, 0.0]


def _a_process_with_faster_whisper_in_it(monkeypatch, whisper_model=lambda *args, **options: _Model()):
    # A copy, so whatever the loader leaves in the table of modules goes when the test does.
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    engine = types.ModuleType("faster_whisper")
    engine.WhisperModel = whisper_model
    sys.modules["faster_whisper"] = engine
    sys.modules.pop("torch", None)


def test_a_model_already_on_this_machine_loads_without_asking_the_hub_for_a_newer_one(monkeypatch):
    asked = []
    _a_process_with_faster_whisper_in_it(
        monkeypatch, lambda *_args, **options: asked.append(options.get("local_files_only")) or _Model())

    load_faster_whisper("base")

    assert asked == [True]


def test_a_model_this_machine_has_never_fetched_is_downloaded(monkeypatch):
    asked = []

    def whisper_model(*_args, **options):
        asked.append(options.get("local_files_only", False))
        if options.get("local_files_only"):
            raise FileNotFoundError("not in the cache")
        return _Model()

    _a_process_with_faster_whisper_in_it(monkeypatch, whisper_model)

    load_faster_whisper("base")

    assert asked == [True, False]


def test_loading_whisper_refuses_the_torch_it_would_otherwise_import(monkeypatch):
    _a_process_with_faster_whisper_in_it(monkeypatch)

    load_faster_whisper("base")

    assert sys.modules.get("torch", "absent") is None


def test_a_torch_something_else_already_imported_is_left_alone(monkeypatch):
    _a_process_with_faster_whisper_in_it(monkeypatch)
    already = types.ModuleType("torch")
    sys.modules["torch"] = already

    load_faster_whisper("base")

    assert sys.modules["torch"] is already
