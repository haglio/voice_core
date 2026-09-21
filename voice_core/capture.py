from __future__ import annotations

import array


class CaptureLevel:
    def __init__(self) -> None:
        self._utterance = 0
        self._recent = 0

    def note_block(self, pcm: bytes) -> None:
        block = array.array("h")
        block.frombytes(pcm[: len(pcm) // 2 * 2])
        if not block:
            return
        peak = max(*block, -min(block))
        self._utterance = max(self._utterance, peak)
        self._recent = max(self._recent, peak)

    def take_utterance(self) -> int:
        peak, self._utterance = self._utterance, 0
        return peak

    def take_recent(self) -> int:
        peak, self._recent = self._recent, 0
        return peak


# Blocks arrive several times a second, so ten seconds of nothing is the stream gone.
AUDIO_STALL_S = 10.0


class AudioStall:
    def __init__(self, *, now: float) -> None:
        self._last_block_at = now
        self._stalled = False

    def note_silence(self, *, now: float) -> float | None:
        idle = now - self._last_block_at
        if self._stalled or idle < AUDIO_STALL_S:
            return None
        self._stalled = True
        return idle

    def note_block(self, *, now: float) -> bool:
        recovered, self._stalled = self._stalled, False
        self._last_block_at = now
        return recovered
