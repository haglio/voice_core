from __future__ import annotations

import array

import numpy as np
import pytest

from voice_core.speech import (
    CONTEXT_SAMPLES,
    SPEECH_GOES_ON_AT,
    SPEECH_STARTS_AT,
    WINDOW_SAMPLES,
    HearsSpeech,
    SileroWindows,
)


def _samples(level: int, count: int) -> bytes:
    return array.array("h", [level] * count).tobytes()


A_WINDOW = _samples(100, WINDOW_SAMPLES)


def _scored(*scores):
    scores = iter(scores)
    return lambda window: next(scores)


def test_nothing_is_speech_until_a_whole_window_of_it_has_been_scored():
    hears = HearsSpeech(_scored(0.9))

    assert [hears(_samples(100, 480)), hears(_samples(100, 480))] == [False, True]


def test_a_window_is_scored_whole_across_the_frames_it_arrived_in():
    windows = []
    hears = HearsSpeech(lambda window: windows.append(window) or 0.0)

    hears(_samples(16384, 480))
    hears(_samples(16384, 480))

    [window] = windows
    assert window == pytest.approx(np.full(WINDOW_SAMPLES, 0.5))


def test_speech_starts_only_at_a_window_scoring_its_bar():
    hears = HearsSpeech(_scored(SPEECH_STARTS_AT - 0.01, SPEECH_STARTS_AT))

    assert [hears(A_WINDOW), hears(A_WINDOW)] == [False, True]


def test_speech_goes_on_through_windows_scoring_the_lower_bar_and_stops_under_it():
    hears = HearsSpeech(_scored(0.9, SPEECH_GOES_ON_AT, SPEECH_GOES_ON_AT - 0.01))

    assert [hears(A_WINDOW) for _ in range(3)] == [True, True, False]


class _Session:
    def __init__(self, score=0.7):
        self.score = score
        self.fed = []

    def run(self, _outputs, feeds):
        self.fed.append(feeds)
        return np.array([self.score], dtype=np.float32), feeds["h"] + 1, feeds["c"] + 2


def test_the_score_of_a_window_is_the_models():
    assert SileroWindows(load=lambda: _Session(0.7))(np.zeros(WINDOW_SAMPLES)) == pytest.approx(0.7)


def test_each_window_is_scored_with_the_end_of_the_one_before_it_in_front():
    session = _Session()
    windows = SileroWindows(load=lambda: session)
    first = np.linspace(0.0, 1.0, WINDOW_SAMPLES, dtype=np.float32)

    windows(first)
    windows(np.zeros(WINDOW_SAMPLES, dtype=np.float32))

    assert session.fed[0]["input"][0][:CONTEXT_SAMPLES] == pytest.approx(np.zeros(CONTEXT_SAMPLES))
    assert session.fed[1]["input"][0][:CONTEXT_SAMPLES] == pytest.approx(first[-CONTEXT_SAMPLES:])


def test_what_the_model_kept_from_one_window_is_handed_back_with_the_next():
    session = _Session()
    windows = SileroWindows(load=lambda: session)

    windows(np.zeros(WINDOW_SAMPLES))
    windows(np.zeros(WINDOW_SAMPLES))

    assert (session.fed[1]["h"], session.fed[1]["c"]) == (pytest.approx(np.ones((1, 1, 128))),
                                                         pytest.approx(np.full((1, 1, 128), 2.0)))


def test_the_model_is_loaded_once_at_the_first_window():
    loads = []
    windows = SileroWindows(load=lambda: loads.append(1) or _Session())

    windows(np.zeros(WINDOW_SAMPLES))
    windows(np.zeros(WINDOW_SAMPLES))

    assert loads == [1]
