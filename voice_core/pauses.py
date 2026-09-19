from __future__ import annotations

from voice_core.microphone import loudness

# How fast the room's level follows the quiet frames: slowly enough that a pause
# inside a sentence does not pull the bar down onto the speaker.
_ROOM_FOLLOWS = 0.02


class PauseSegmenter:
    """Cuts a stream of frames into utterances where the speaker pauses.

    For speech no phrase list holds: vosk ends an utterance when its grammar has
    nothing more to say, which for ordinary talk is twenty seconds late or never.
    """

    def __init__(self, *, floor: float, ratio: float, calibration_frames: int,
                 hangover_frames: int, min_speech_frames: int) -> None:
        self._floor = floor
        self._ratio = ratio
        self._calibrating = calibration_frames
        self._calibration: list[float] = []
        self._room: float | None = None
        self._hangover_frames = hangover_frames
        self._min_speech_frames = min_speech_frames
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
            return None
        self._room += (level - self._room) * _ROOM_FOLLOWS
        if not self._frames:
            return None
        self._frames.append(frame)
        self._quiet_run += 1
        if self._quiet_run < self._hangover_frames:
            return None
        utterance, spoken = b"".join(self._frames), self._speech_frames
        self._frames, self._speech_frames, self._quiet_run = [], 0, 0
        return utterance if spoken >= self._min_speech_frames else None
