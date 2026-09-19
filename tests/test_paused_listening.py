from __future__ import annotations

import array
import json

from voice_core.commands import CommandRules, Recognition
from voice_core.listening import Heard, PausedListening, Recognizers
from voice_core.pauses import PauseSegmenter

RULES = CommandRules(phrases=frozenset({"next", "left next"}), never_rescued=lambda phrase: False)
RATE = 16000
FRAME = 480
LOUD = 16000
QUIET = array.array("h", [0] * FRAME).tobytes()
VOICED = array.array("h", [LOUD, -LOUD] * (FRAME // 2)).tobytes()
A_FRAME = FRAME / RATE


class _Recognizer:
    """Settles by itself on the frames it was told to; asked to finish, gives what it was given."""

    def __init__(self, finish_with, settles_at=()):
        self._finish_with = finish_with
        self._settles_at = dict(settles_at)
        self._fed = 0
        self.finished_after = []

    def AcceptWaveform(self, data):  # noqa: N802 -- vosk's spelling
        self._fed += 1
        return self._fed in self._settles_at

    def Result(self):  # noqa: N802
        return self._settles_at[self._fed]

    def FinalResult(self):  # noqa: N802
        self.finished_after.append(self._fed)
        return self._finish_with

    def PartialResult(self):  # noqa: N802
        return json.dumps({"partial": ""})


def _ranked(*texts):
    return json.dumps({"alternatives": [{"text": text, "confidence": 1.0} for text in texts]})


def _segmenter():
    segmenter = PauseSegmenter(floor=1600, ratio=1.5, calibration_frames=1, hangover_frames=2,
                               min_speech_frames=2)
    segmenter.push(QUIET)
    return segmenter


def _feed(listening, frames, first_captured_at=10.0):
    heard = [listening.feed(frame, captured_at=first_captured_at + number * A_FRAME)
             for number, frame in enumerate(frames)]
    return [one for one in heard if one is not None]


def test_an_utterance_ends_where_the_speaker_pauses_and_the_recognizer_is_made_to_finish_there():
    listening = PausedListening(RULES, Recognizers(_Recognizer(_ranked("left next"))),
                                _segmenter(), sample_rate=RATE)

    heard = _feed(listening, [VOICED, VOICED, QUIET, QUIET])

    assert heard == [Heard(Recognition(phrase="left next", heard="left next"),
                           spoken_at=10.0 - A_FRAME, peak=LOUD,
                           audio=VOICED + VOICED + QUIET + QUIET, candidates={"left next": 0})]


def test_an_utterance_the_recognizer_settled_partway_through_is_more_than_one_command_can_be():
    recognizer = _Recognizer(_ranked("next"), settles_at={2: _ranked("left next")})
    listening = PausedListening(RULES, Recognizers(recognizer), _segmenter(), sample_rate=RATE)

    [heard] = _feed(listening, [VOICED, VOICED, VOICED, VOICED, QUIET, QUIET])

    assert heard.recognition == Recognition(unrecognized_text="left next next",
                                            heard="left next next")
    assert heard.candidates == {}


def test_what_the_recognizer_made_of_the_quiet_between_utterances_is_not_carried_into_the_next():
    recognizer = _Recognizer(_ranked("next"), settles_at={1: _ranked("left next")})
    listening = PausedListening(RULES, Recognizers(recognizer), _segmenter(), sample_rate=RATE)

    [heard] = _feed(listening, [QUIET, VOICED, VOICED, QUIET, QUIET])

    assert heard.recognition == Recognition(phrase="next", heard="next")


def test_a_sound_too_short_to_be_a_word_is_cleared_out_of_the_recognizer_as_well():
    recognizer = _Recognizer(_ranked("next"))
    listening = PausedListening(RULES, Recognizers(recognizer), _segmenter(), sample_rate=RATE)

    heard = _feed(listening, [VOICED, QUIET, QUIET, VOICED, VOICED, QUIET, QUIET])

    assert len(heard) == 1
    assert recognizer.finished_after == [3, 7]
