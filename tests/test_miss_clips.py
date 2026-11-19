from __future__ import annotations

import wave
from datetime import UTC, datetime

from voice_core.miss_clips import MissClips


def _at(microsecond):
    return datetime(2026, 1, 2, 3, 4, 5, microsecond, tzinfo=UTC)


def _right_away(job):
    job()


def test_a_miss_is_kept_as_a_wav_named_for_when_it_happened(tmp_path):
    clips = tmp_path / "voice_misses"

    MissClips(clips, sample_rate=16000, now=lambda: _at(6), later=_right_away).keep(bytes([1, 0]) * 16)

    [path] = clips.iterdir()
    assert path == clips / "2026-01-02_03-04-05_000006.wav"
    with wave.open(str(path), "rb") as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate(),
                clip.getnframes()) == (1, 2, 16000, 16)


def test_only_the_newest_clips_are_kept_counting_those_already_in_the_folder(tmp_path):
    clips = tmp_path / "voice_misses"
    times = iter(range(4))
    MissClips(clips, sample_rate=16000, keep=2, now=lambda: _at(next(times)),
              later=_right_away).keep(b"\x01\x00")
    misses = MissClips(clips, sample_rate=16000, keep=2, now=lambda: _at(next(times)),
                       later=_right_away)

    for _ in range(3):
        misses.keep(b"\x01\x00")

    assert [path.stem[-1] for path in sorted(clips.iterdir())] == ["2", "3"]


def test_a_clip_is_written_after_the_miss_is_handed_on_never_while_it_waits(tmp_path):
    waiting = []
    clips = tmp_path / "voice_misses"
    misses = MissClips(clips, sample_rate=16000, now=lambda: _at(6), later=waiting.append)

    misses.keep(b"\x01\x00")
    written_at_once = clips.exists()
    waiting.pop()()

    assert (written_at_once, [path.name for path in clips.iterdir()]) == (
        False, ["2026-01-02_03-04-05_000006.wav"])


def test_closing_waits_for_the_clips_still_being_written_and_writing_goes_on_after(tmp_path):
    clips = tmp_path / "voice_misses"
    misses = MissClips(clips, sample_rate=16000)

    misses.keep(b"\x01\x00")
    misses.close()
    written = len(list(clips.iterdir()))
    misses.keep(b"\x02\x00")
    misses.close()

    assert (written, len(list(clips.iterdir()))) == (1, 2)


def test_a_clip_that_cannot_be_written_is_logged_and_listening_goes_on(tmp_path, caplog):
    in_the_way = tmp_path / "a file where the folder should be"
    in_the_way.write_text("")

    MissClips(in_the_way, sample_rate=16000, later=_right_away).keep(b"\x01\x00")

    assert "could not be kept" in caplog.text
