from __future__ import annotations

import array
import json
from contextlib import contextmanager
from types import SimpleNamespace

from voice_core.listener import Engines
from voice_core.pauses import PauseSettings

_PAUSES = PauseSettings()
LOUD_FRAME = array.array("h", [2000, -2000] * (_PAUSES.frame_samples // 2)).tobytes()
QUIET_FRAME = bytes(_PAUSES.frame_samples * 2)


class Microphone(SimpleNamespace):
    def __init__(self, frames):
        devices = [{"name": "Desk mic", "max_input_channels": 1, "hostapi": 0}]
        self.opened: list[dict] = []
        self.closed = 0

        @contextmanager
        def stream(**kwargs):
            self.opened.append(kwargs)
            for frame in frames:
                kwargs["callback"](frame, len(frame) // 2, None, None)
            try:
                yield
            finally:
                self.closed += 1

        super().__init__(
            default=SimpleNamespace(device=(0, 0)),
            query_devices=lambda index=None: devices if index is None else devices[index],
            RawInputStream=stream)


def vosk(recognizer) -> SimpleNamespace:
    return SimpleNamespace(Model=lambda model_name: model_name, KaldiRecognizer=recognizer)


def said(*readings: str, forming=()) -> tuple[list[str], list[str]]:
    return list(forming), list(readings)


class ScriptedRecognizer:
    def __init__(self, utterances):
        self._utterances = list(utterances)
        self._forming: list[str] = []
        self._readings: list[str] = []
        self._partial = ""

    def SetWords(self, enable):  # noqa: N802 - vosk's own name
        pass

    def SetMaxAlternatives(self, count):  # noqa: N802 - vosk's own name
        pass

    def Reset(self):  # noqa: N802 - vosk's own name
        self._forming, self._readings = self._utterances.pop(0) if self._utterances else ([], [])
        self._partial = ""

    def AcceptWaveform(self, _frame):  # noqa: N802 - vosk's own name
        if self._forming:
            self._partial = self._forming.pop(0)
        return False

    def PartialResult(self):  # noqa: N802 - vosk's own name
        return json.dumps({"partial": self._partial})

    def FinalResult(self):  # noqa: N802 - vosk's own name
        return json.dumps({"alternatives": [
            {"text": reading, "confidence": 1.0} for reading in self._readings]})


def frames_saying(utterances, pauses: PauseSettings = _PAUSES) -> list[bytes]:
    frames = [QUIET_FRAME] * pauses.calibration_frames
    for forming, _ in utterances:
        frames += [LOUD_FRAME] * max(len(forming), pauses.min_speech_frames)
        frames += [QUIET_FRAME] * pauses.hangover_frames
    return frames


def scripted_engines(utterances, *, frames=None, **more) -> Engines:
    return Engines(
        vosk=vosk(lambda *args: ScriptedRecognizer(utterances)),
        sounddevice=Microphone(frames_saying(utterances) if frames is None else frames),
        **more)
