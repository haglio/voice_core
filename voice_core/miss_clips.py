from __future__ import annotations

import wave
from datetime import datetime
from pathlib import Path

MISS_CLIPS_KEPT = 500


def save_miss_audio(directory: Path, pcm: bytes, *, sample_rate: int,
                    keep: int = MISS_CLIPS_KEPT, now: datetime | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H-%M-%S_%f")
    path = directory / f"{stamp}.wav"
    with wave.open(str(path), "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(sample_rate)
        clip.writeframes(pcm)
    for stale in sorted(directory.glob("*.wav"))[:-keep]:
        stale.unlink()
    return path
