from __future__ import annotations

import array
import json
import logging

import pytest

from voice_core.commands import CommandRules, Recognition
from voice_core.listening import Heard, Listening, Recognizers, outcome_line
from voice_core.pauses import PauseSegmenter, PauseSettings

RULES = CommandRules(phrases=frozenset({"next", "left next"}), never_rescued=lambda phrase: False)
RATE = 16000
FRAME = 480
A_FRAME = FRAME / RATE
LOUD = 2000
SHOUT = 9000
VOICED = array.array("h", [LOUD, -LOUD] * (FRAME // 2)).tobytes()
SHOUTED = array.array("h", [SHOUT, -SHOUT] * (FRAME // 2)).tobytes()
QUIET = bytes(FRAME * 2)
PAUSES = PauseSettings(floor=1600, ratio=1.5, calibration_frames=1, hangover_frames=2,
                       min_speech_frames=2)
FIRST_FRAME_AT = 10.0
A_COMMAND = [QUIET, VOICED, VOICED, QUIET, QUIET]
SPOKEN_AT = FIRST_FRAME_AT + A_FRAME


class _Recognizer:
    def __init__(self, finish_with="", settles_at=(), partials=()):
        self._finish_with = finish_with
        self._settles_at = dict(settles_at)
        self._partials = dict(partials)
        self.fed = 0
        self.heard = []
        self.cleared_after = []

    def AcceptWaveform(self, data):  # noqa: N802 -- vosk's spelling
        self.fed += 1
        self.heard.append(data)
        return self.fed in self._settles_at

    def Result(self):  # noqa: N802
        return self._settles_at[self.fed]

    def FinalResult(self):  # noqa: N802
        return self._finish_with

    def PartialResult(self):  # noqa: N802
        return json.dumps({"partial": self._partials.get(self.fed, "")})

    def Reset(self):  # noqa: N802
        self.cleared_after.append(self.fed)


def _ranked(*texts):
    return json.dumps({"alternatives": [{"text": text, "confidence": 1.0} for text in texts]})


def _scored(text, conf):
    return json.dumps({"text": text, "result": [{"conf": conf, "word": w} for w in text.split()]})


def _listening(recognizer, unrestricted=None, **how):
    return Listening(RULES, Recognizers(recognizer, unrestricted), PauseSegmenter(PAUSES), **how)


def _feed(listening, frames):
    heard = [listening.feed(frame, started_at=FIRST_FRAME_AT + number * A_FRAME)
             for number, frame in enumerate(frames)]
    return [one for one in heard if one is not None]


def test_an_utterance_ends_where_the_speaker_pauses_and_the_recognizer_finishes_there():
    listening = _listening(_Recognizer(_ranked("left next")))

    heard = _feed(listening, A_COMMAND)

    assert heard == [Heard(Recognition(phrase="left next", heard="left next"),
                           spoken_at=SPOKEN_AT, peak=LOUD,
                           audio=VOICED + VOICED + QUIET + QUIET, candidates={"left next": 0})]


def test_what_the_recognizer_read_of_the_room_is_cleared_where_the_speaker_starts():
    recognizer = _Recognizer(_ranked("next"))

    _feed(_listening(recognizer), A_COMMAND)

    assert recognizer.cleared_after == [1]


def test_an_utterance_is_dated_from_the_frame_the_speaker_was_first_heard_on():
    [heard] = _feed(_listening(_Recognizer(_ranked("next"))), [QUIET, QUIET, *A_COMMAND[1:]])

    assert heard.spoken_at == pytest.approx(SPOKEN_AT + A_FRAME)


def test_the_words_forming_are_passed_on_as_they_change_while_the_speaker_is_heard():
    forming = []
    recognizer = _Recognizer(_ranked("left next"), partials={2: "left", 3: "left next"})

    _feed(_listening(recognizer, on_partial=forming.append), A_COMMAND)

    assert forming == ["left", "left next", ""]


def test_nothing_the_recognizer_reads_out_of_the_quiet_room_is_called_words_forming():
    forming = []
    recognizer = _Recognizer(partials=dict.fromkeys(range(1, 9), "net"))

    _feed(_listening(recognizer, on_partial=forming.append), [QUIET] * 8)

    assert forming == []


def test_a_sound_too_short_to_be_a_word_is_no_part_of_the_utterance_after_it():
    [heard] = _feed(_listening(_Recognizer(_ranked("next"))),
                    [QUIET, SHOUTED, QUIET, QUIET, VOICED, VOICED, QUIET, QUIET])

    assert (heard.peak, heard.audio) == (LOUD, VOICED + VOICED + QUIET + QUIET)


def test_a_command_the_recognizer_settled_partway_through_is_still_the_command():
    recognizer = _Recognizer(settles_at={3: _ranked("left next")})

    [heard] = _feed(_listening(recognizer), A_COMMAND)

    assert heard.recognition == Recognition(phrase="left next", heard="left next")


def test_several_readings_none_of_them_a_command_are_heard_as_talk():
    recognizer = _Recognizer(_ranked("left net"), settles_at={3: _ranked("net")})

    [heard] = _feed(_listening(recognizer), A_COMMAND)

    assert heard.recognition == Recognition(unrecognized_text="net left net",
                                            heard="net left net")
    assert heard.candidates == {}


def test_for_an_app_that_takes_dictation_several_readings_are_more_than_one_command_can_be():
    recognizer = _Recognizer(_ranked("next"), settles_at={3: _ranked("left next")})

    [heard] = _feed(_listening(recognizer, for_dictation=True), A_COMMAND)

    assert heard.recognition == Recognition(unrecognized_text="left next next",
                                            heard="left next next")


def test_a_miss_carries_what_an_unrestricted_recognizer_made_of_the_same_utterance():
    unrestricted = _Recognizer(_scored("what is next", 0.4))
    listening = _listening(_Recognizer(_ranked("left net")), unrestricted)

    [heard] = _feed(listening, A_COMMAND)

    assert heard.recognition == Recognition(
        unrecognized_text="left net", heard="left net", free_text="what is next")


def test_what_the_unrestricted_recognizer_settled_by_itself_is_in_the_caption_too():
    unrestricted = _Recognizer(_scored("next", 0.4), settles_at={1: _scored("what is", 0.4)})
    listening = _listening(_Recognizer(_ranked("left net")), unrestricted)

    [heard] = _feed(listening, A_COMMAND)

    assert heard.recognition.free_text == "what is next"


def test_a_command_is_never_held_up_by_the_unrestricted_recognizer():
    unrestricted = _Recognizer(_scored("next", 0.9))

    [heard] = _feed(_listening(_Recognizer(_ranked("next")), unrestricted), A_COMMAND)

    assert heard.recognition.phrase == "next"
    assert unrestricted.heard == []


def test_a_miss_is_captioned_from_its_own_audio_read_in_one_go():
    unrestricted = _Recognizer(_scored("what", 0.4))

    _feed(_listening(_Recognizer(_ranked("left net")), unrestricted), [QUIET] * 6 + A_COMMAND[1:])

    assert unrestricted.heard == [VOICED + VOICED + QUIET + QUIET]


def _heard(recognition):
    return Heard(recognition, spoken_at=10.0, peak=LOUD, audio=b"", candidates={})


def test_a_command_is_logged_with_how_long_ago_it_was_spoken_and_how_loud():
    level, line = outcome_line(_heard(Recognition(phrase="next", heard="next")), now=11.25,
                               rules=RULES)

    assert (level, line) == (
        logging.INFO, "Voice command: 'next' (spoken 1.25s before recognition, peak 2000)")


def test_a_repair_is_logged_beside_the_reading_it_was_taken_from_under():
    _, line = outcome_line(_heard(Recognition(phrase="left next", rank=1, heard="left net")),
                           now=11.0, rules=RULES)

    assert line == ("Voice command: 'left next' -- the recognizer's choice 2, under 'left net', "
                    "which is no command (spoken 1.00s before recognition, peak 2000)")


def test_every_way_of_not_being_a_command_is_logged_under_its_own_name():
    lines = [outcome_line(_heard(recognition), now=11.0, rules=RULES) for recognition in (
        Recognition(refused_phrase="next", heard="next", free_text="text"),
        Recognition(unrecognized_text="left net", heard="left net"),
        Recognition(silent_reading="half"),
        Recognition(),
    )]

    assert lines == [
        (logging.INFO, ("Voice: heard 'next' but its confidence was under 0.70 "
                        "(unrestricted reading 'text', peak 2000)")),
        (logging.INFO, "Unrecognized speech: 'left net' (unrestricted reading '', peak 2000)"),
        (logging.INFO, "Voice: ignored 'half' read from silence (peak 2000)"),
        (logging.DEBUG, "Voice: an utterance ended with nothing in it (peak 2000)"),
    ]


def test_a_command_the_second_engine_did_not_read_is_logged_beside_what_it_read_instead():
    doubted = Recognition(unconfirmed_phrase="next", heard="next", free_text="and then")

    assert outcome_line(_heard(doubted), now=11.0, rules=RULES) == (
        logging.INFO, "Voice: heard 'next' but the second engine read 'and then' (peak 2000)")
