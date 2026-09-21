from __future__ import annotations

import array

from voice_core.capture import AUDIO_STALL_S, AudioStall, CaptureLevel


def _pcm(*samples):
    return array.array("h", samples).tobytes()


LOUD = 900
FAINT = 20


def test_the_level_of_an_utterance_is_its_loudest_sample_either_side_of_zero():
    level = CaptureLevel()
    level.note_block(_pcm(10, -LOUD, 30))
    level.note_block(_pcm(400, 5))

    assert level.take_utterance() == LOUD


def test_taking_an_utterances_level_starts_the_next_one_from_nothing():
    level = CaptureLevel()
    level.note_block(_pcm(LOUD))
    level.take_utterance()
    level.note_block(_pcm(-FAINT))

    assert level.take_utterance() == FAINT


def test_the_level_since_the_last_report_is_kept_apart_from_the_utterances():
    level = CaptureLevel()
    level.note_block(_pcm(LOUD))
    level.take_utterance()
    level.note_block(_pcm(-FAINT))

    assert level.take_recent() == LOUD
    assert level.take_recent() == 0


def test_a_block_with_no_whole_sample_in_it_changes_nothing():
    level = CaptureLevel()
    level.note_block(b"\x01")

    assert level.take_utterance() == 0


def test_a_steady_stream_is_never_a_stall():
    stall = AudioStall(now=0.0)

    assert stall.note_silence(now=AUDIO_STALL_S - 1) is None
    assert stall.note_block(now=AUDIO_STALL_S - 0.5) is False


def test_a_stall_is_reported_once_with_how_long_nothing_has_arrived():
    stall = AudioStall(now=0.0)

    assert stall.note_silence(now=AUDIO_STALL_S) == AUDIO_STALL_S
    assert stall.note_silence(now=AUDIO_STALL_S + 5) is None


def test_the_first_block_after_a_stall_says_the_audio_is_back_and_a_new_stall_can_be_told():
    stall = AudioStall(now=0.0)
    stall.note_silence(now=AUDIO_STALL_S)

    assert stall.note_block(now=AUDIO_STALL_S + 1) is True
    assert stall.note_block(now=AUDIO_STALL_S + 2) is False
    assert stall.note_silence(now=2 * AUDIO_STALL_S + 2) == AUDIO_STALL_S
