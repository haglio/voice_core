from __future__ import annotations

import array

from voice_core.pauses import PauseSegmenter, PauseSettings

SILENCE = array.array("h", [0] * 4).tobytes()
SPEECH = array.array("h", [16000, -16000] * 2).tobytes()


def _settings(**settings):
    return PauseSettings(**{"floor": 1600, "ratio": 1.5, "calibration_frames": 2,
                            "hangover_frames": 3, "min_speech_frames": 2, **settings})


def _calibrated(**settings):
    segmenter = PauseSegmenter(_settings(**settings))
    segmenter.push(SILENCE)
    segmenter.push(SILENCE)
    return segmenter


def test_silence_never_ends_an_utterance():
    segmenter = _calibrated()

    assert all(segmenter.push(SILENCE) is None for _ in range(10))


def test_speech_followed_by_a_long_enough_pause_is_an_utterance_with_its_trailing_quiet():
    segmenter = _calibrated()

    ended = [segmenter.push(frame) for frame in (SPEECH, SPEECH, SILENCE, SILENCE, SILENCE)]

    assert ended[:4] == [None] * 4
    assert ended[4] == SPEECH + SPEECH + SILENCE * 3


def test_a_sound_shorter_than_any_word_is_dropped():
    segmenter = _calibrated()

    ended = [segmenter.push(frame) for frame in (SPEECH, SILENCE, SILENCE, SILENCE)]

    assert ended == [None] * 4


def test_one_utterance_after_another_is_heard_as_two():
    segmenter = _calibrated()
    spoken = (SPEECH, SPEECH, SILENCE, SILENCE, SILENCE)

    ended = [segmenter.push(frame) for frame in spoken + spoken]

    assert [one is not None for one in ended] == [False] * 4 + [True] + [False] * 4 + [True]


ROOM_TONE = array.array("h", [1600, -1600] * 2).tobytes()
A_LITTLE_OVER_THE_ROOM = array.array("h", [2000, -2000] * 2).tobytes()


def test_the_bar_for_speech_is_a_multiple_of_the_rooms_own_level_measured_as_listening_starts():
    segmenter = PauseSegmenter(PauseSettings(floor=10, ratio=2.0, calibration_frames=3,
                                             hangover_frames=3, min_speech_frames=2))
    for _ in range(3):
        assert segmenter.push(ROOM_TONE) is None

    ended = [segmenter.push(frame) for frame in
             (A_LITTLE_OVER_THE_ROOM, A_LITTLE_OVER_THE_ROOM, ROOM_TONE, ROOM_TONE, ROOM_TONE,
              SPEECH, SPEECH, ROOM_TONE, ROOM_TONE, ROOM_TONE)]

    assert [one is not None for one in ended] == [False] * 9 + [True]


def test_the_bar_never_drops_under_the_floor_however_quiet_the_room():
    segmenter = _calibrated(floor=5000)

    ended = [segmenter.push(frame) for frame in
             (A_LITTLE_OVER_THE_ROOM, A_LITTLE_OVER_THE_ROOM, SILENCE, SILENCE, SILENCE)]

    assert ended == [None] * 5


def test_a_sound_that_never_pauses_is_cut_where_it_has_gone_on_too_long():
    segmenter = _calibrated(longest_frames=4)

    ended = [segmenter.push(SPEECH) for _ in range(9)]

    assert [one is not None for one in ended] == [False, False, False, True] * 2 + [False]
    assert ended[3] == SPEECH * 4


def test_the_room_a_session_starts_in_is_no_longer_the_bar_once_it_has_gone_quiet():
    segmenter = PauseSegmenter(PauseSettings(floor=10, ratio=2.0, calibration_frames=1,
                                             hangover_frames=3, min_speech_frames=2))
    segmenter.push(ROOM_TONE)
    for _ in range(400):
        segmenter.push(SILENCE)

    ended = [segmenter.push(frame) for frame in
             (A_LITTLE_OVER_THE_ROOM, A_LITTLE_OVER_THE_ROOM, SILENCE, SILENCE, SILENCE)]

    assert [one is not None for one in ended] == [False] * 4 + [True]


def test_it_says_whether_somebody_is_in_the_middle_of_speaking():
    segmenter = _calibrated()
    states = []
    for frame in (SILENCE, SPEECH, SPEECH, SILENCE, SILENCE, SILENCE):
        segmenter.push(frame)
        states.append(segmenter.speaking)

    assert states == [False, True, True, True, True, False]
