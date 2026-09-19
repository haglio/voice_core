from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

# Prompted with an utterance's own candidates, "base" chose as well as "small" on the owner's
# recordings (7 of 10 missed commands either way, 3 against 9 invented in 930 utterances of
# ordinary talk) in a quarter of the time: about a third of a second a reading on this machine.
COMMAND_MODEL = "base"

# Taking his sentences down is where "small" earns its second: measured against the engine his
# dictation apps use, its worst quarter of 24 utterances came within 0.86 of it, base within 0.59.
DICTATION_MODEL = "small"

_FULL_LEVEL = 0.95
_SILENCE = 1e-4


def load_faster_whisper(model_size: str):
    # faster-whisper needs no torch but imports any it finds, and a torch that loads
    # fine alone can die initializing c10.dll once Qt's DLLs are in the process
    # (WinError 1114), taking every reading with it. A torch already imported stays.
    sys.modules.setdefault("torch", None)
    from faster_whisper import WhisperModel  # noqa: PLC0415 -- heavy, and only an app that asks

    # On the CPU by design: the one GPU is for pictures and players. A reading took a median
    # 0.43 s on two threads, 0.33 s on four and 0.27 s on eight (this machine, 2026-09-18):
    # four is most of the gain and leaves the players their cores.
    return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)


class WhisperReader:
    """Reads one utterance: ``reader(pcm, hint)``.

    As recorded, for choosing among a command's candidates -- lifting the level or filtering
    for speech cost that reading a clip or two in ten. *for_dictation*, a quiet microphone is
    lifted to full level and whisper skips what is not speech, which is what steadied sentences.
    """

    def __init__(self, *, load: Callable[[], Any] | None = None, for_dictation: bool = False) -> None:
        size = DICTATION_MODEL if for_dictation else COMMAND_MODEL
        self._load = load or partial(load_faster_whisper, size)
        self._for_dictation = for_dictation
        self._model = None

    def preload(self) -> None:
        if self._model is None:
            self._model = self._load()

    def __call__(self, pcm: bytes, hint: str) -> str:
        self.preload()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self._for_dictation:
            audio = audio / max(float(np.max(np.abs(audio), initial=0.0)), _SILENCE) * _FULL_LEVEL
        segments, _ = self._model.transcribe(
            audio, language="en", initial_prompt=hint or None,
            vad_filter=self._for_dictation, condition_on_previous_text=False)
        return " ".join(part for part in (segment.text.strip() for segment in segments) if part)
