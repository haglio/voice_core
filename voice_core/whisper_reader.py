from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

# Prompted with an utterance's own candidates, "base" chose as well as "small" on the owner's
# recordings (7 of 10 missed commands either way, 3 against 9 invented in 930 utterances of
# ordinary talk) in a quarter of the time: about half a second a reading on this machine.
MODEL_SIZE = "base"


def load_faster_whisper(model_size: str = MODEL_SIZE):
    # faster-whisper needs no torch but imports any it finds, and a torch that loads
    # fine alone can die initializing c10.dll once Qt's DLLs are in the process
    # (WinError 1114), taking every reading with it. A torch already imported stays.
    sys.modules.setdefault("torch", None)
    from faster_whisper import WhisperModel  # noqa: PLC0415 -- heavy, and only an app that asks

    # On the CPU by design: the one GPU is for pictures and players.
    return WhisperModel(model_size, device="cpu", compute_type="int8")


class WhisperReader:
    def __init__(self, *, load: Callable[[], Any] = load_faster_whisper) -> None:
        self._load = load
        self._model = None

    def preload(self) -> None:
        if self._model is None:
            self._model = self._load()

    def __call__(self, pcm: bytes, hint: str) -> str:
        self.preload()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio, language="en", initial_prompt=hint or None, condition_on_previous_text=False)
        return " ".join(part for part in (segment.text.strip() for segment in segments) if part)
