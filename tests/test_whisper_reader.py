from __future__ import annotations

import array
from types import SimpleNamespace

import pytest

from voice_core.whisper_reader import WhisperReader

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


def test_with_no_hint_whisper_is_given_no_prompt_at_all():
    model = _Model("anything at all")

    WhisperReader(load=lambda: model)(QUIET_PCM, "")

    assert model.asked[0][1]["initial_prompt"] is None
