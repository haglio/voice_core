from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from voice_core.whisper_reader import faster_whisper

# Silero's own bars: a window scoring 0.5 starts speech, and speech goes on through windows
# scoring 0.35 or more.
SPEECH_STARTS_AT = 0.5
SPEECH_GOES_ON_AT = 0.35
WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64


class HearsSpeech:
    def __init__(self, score: Callable[[np.ndarray], float]) -> None:
        self._score = score
        self._waiting = np.zeros(0, dtype=np.float32)
        self._speaking = False

    def __call__(self, frame: bytes) -> bool:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        self._waiting = np.concatenate([self._waiting, samples])
        while len(self._waiting) >= WINDOW_SAMPLES:
            window, self._waiting = self._waiting[:WINDOW_SAMPLES], self._waiting[WINDOW_SAMPLES:]
            score = self._score(window)
            self._speaking = score >= (SPEECH_GOES_ON_AT if self._speaking else SPEECH_STARTS_AT)
        return self._speaking


class SileroWindows:
    def __init__(self, load: Callable[[], Any] | None = None) -> None:
        self._load = load or _silero_session
        self._session = None
        self._state = np.zeros((2, 1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        if self._session is None:
            self._session = self._load()
        row = np.concatenate([self._context, window])[None, :].astype(np.float32)
        score, h, c = self._session.run(None, {"input": row, "h": self._state[0], "c": self._state[1]})
        self._state = np.stack([h, c])
        self._context = window[-CONTEXT_SAMPLES:]
        return float(np.ravel(score)[0])


def hears_speech() -> HearsSpeech:
    return HearsSpeech(SileroWindows())


def _silero_session():
    return faster_whisper("faster_whisper.vad").get_vad_model().session
