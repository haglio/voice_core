from __future__ import annotations

import logging
import wave
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

logger = logging.getLogger(__name__)

MISS_CLIPS_KEPT = 500

_ONE_WRITER = ThreadPoolExecutor(1, thread_name_prefix="voice-miss-clips")


def _now() -> datetime:
    return datetime.now().astimezone()


class MissClips:
    def __init__(self, directory: Path, *, sample_rate: int, keep: int = MISS_CLIPS_KEPT,
                 now: Callable[[], datetime] = _now,
                 later: Callable[[Callable[[], None]], object] | None = None) -> None:
        self._directory = directory
        self._sample_rate = sample_rate
        self._keep = keep
        self._now = now
        self._later = later or _ONE_WRITER.submit
        self._kept: list[Path] | None = None
        self._last = None

    def keep(self, pcm: bytes) -> None:
        self._last = self._later(partial(self._write, pcm, self._now()))

    def close(self) -> None:
        if isinstance(self._last, Future):
            self._last.result()

    def _write(self, pcm: bytes, when: datetime) -> None:
        try:
            self._write_and_trim(pcm, when)
        except OSError:
            logger.warning("Voice: a missed utterance's clip could not be kept", exc_info=True)

    def _write_and_trim(self, pcm: bytes, when: datetime) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{when.strftime('%Y-%m-%d_%H-%M-%S_%f')}.wav"
        with wave.open(str(path), "wb") as clip:
            clip.setnchannels(1)
            clip.setsampwidth(2)
            clip.setframerate(self._sample_rate)
            clip.writeframes(pcm)
        if self._kept is None:
            self._kept = sorted(self._directory.glob("*.wav"))
        else:
            self._kept.append(path)
        while len(self._kept) > self._keep:
            self._kept.pop(0).unlink(missing_ok=True)
