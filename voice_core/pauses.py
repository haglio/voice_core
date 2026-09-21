from __future__ import annotations

from dataclasses import dataclass

from voice_core.microphone import loudness

# How fast the room's level follows the quiet frames: slowly enough that a pause
# inside a sentence does not pull the bar down onto the speaker.
_ROOM_FOLLOWS = 0.02


@dataclass(frozen=True)
class PauseSettings:
    """Where an utterance begins and ends, in 16-bit sample units.

    The owner's microphone is quiet -- his speech runs 60 to 600 a frame, against digital
    silence -- so the floor is the one his dictation app settled on over replayed sessions
    (0.0008 of full scale), not the 0.008 Origenerator had, which sat above most of his words.
    Half a second of quiet ends an utterance; under 180 ms of sound holds no word."""

    floor: float = 26.0
    ratio: float = 2.5
    calibration_frames: int = 15
    hangover_frames: int = 17
    min_speech_frames: int = 6
    frame_samples: int = 480  # 30 ms at 16 kHz
    # Twenty seconds, where vosk's own endpointer gave up on a room that never pauses.
    longest_frames: int = 667


class PauseSegmenter:
    """Vosk's own endpointer cannot end an utterance on the owner's microphone: the noise
    suppressor's silence reads to vosk as words, so it seldom hears the silence it waits for."""

    def __init__(self, settings: PauseSettings) -> None:
        self._floor = settings.floor
        self._ratio = settings.ratio
        self._calibrating = settings.calibration_frames
        self._calibration: list[float] = []
        self._room: float | None = None
        self._hangover_frames = settings.hangover_frames
        self._min_speech_frames = settings.min_speech_frames
        self._longest_frames = settings.longest_frames
        self._frames: list[bytes] = []
        self._speech_frames = 0
        self._quiet_run = 0

    @property
    def speaking(self) -> bool:
        return bool(self._frames)

    def push(self, frame: bytes) -> bytes | None:
        level = loudness(frame)
        if self._room is None:
            self._calibration.append(level)
            if len(self._calibration) >= self._calibrating:
                self._room = sum(self._calibration) / len(self._calibration)
            return None
        if level >= max(self._floor, self._room * self._ratio):
            self._frames.append(frame)
            self._speech_frames += 1
            self._quiet_run = 0
        else:
            self._room += (level - self._room) * _ROOM_FOLLOWS
            if not self._frames:
                return None
            self._frames.append(frame)
            self._quiet_run += 1
        if self._quiet_run >= self._hangover_frames or self._has_gone_on_too_long():
            return self._take()
        return None

    def _has_gone_on_too_long(self) -> bool:
        return len(self._frames) >= self._longest_frames

    def _take(self) -> bytes | None:
        utterance, spoken = b"".join(self._frames), self._speech_frames
        self._frames, self._speech_frames, self._quiet_run = [], 0, 0
        return utterance if spoken >= self._min_speech_frames else None
