from __future__ import annotations

import wave
from datetime import UTC, datetime

from voice_core.miss_clips import save_miss_audio


def test_a_miss_is_kept_as_a_wav_named_for_when_it_happened(tmp_path):
    clips = tmp_path / "voice_misses"

    path = save_miss_audio(clips, bytes([1, 0]) * 16, sample_rate=16000,
                           now=datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=UTC))

    assert path == clips / "2026-01-02_03-04-05_000006.wav"
    with wave.open(str(path), "rb") as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate(),
                clip.getnframes()) == (1, 2, 16000, 16)


def test_only_the_newest_clips_are_kept(tmp_path):
    clips = tmp_path / "voice_misses"

    paths = [save_miss_audio(clips, b"\x01\x00", sample_rate=16000, keep=2,
                             now=datetime(2026, 1, 2, 3, 4, 5, i, tzinfo=UTC))
             for i in range(3)]

    assert sorted(clips.iterdir()) == paths[1:]
