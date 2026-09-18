from __future__ import annotations

import array
import json
import logging

from voice_core.commands import CommandRules, Recognition
from voice_core.listening import Heard, Listening, outcome_line

RULES = CommandRules(phrases=frozenset({"next", "left next"}), never_rescued=lambda phrase: False)
RATE = 16000
HALF_SECOND = RATE // 2
LOUD = 2000
SECOND_BLOCK_BEGAN = 10.5


def _block(level=LOUD, samples=HALF_SECOND):
    return array.array("h", [level, -level] * (samples // 2)).tobytes()


class _ScriptedRecognizer:
    """Finalizes on the blocks it was told to, with the result it was given for each."""

    def __init__(self, finals=(), partials=(), still_forming=""):
        self._finals = dict(finals)
        self._partials = dict(partials)
        self._still_forming = still_forming
        self._fed = 0

    def AcceptWaveform(self, data):  # noqa: N802 -- vosk's spelling
        self._fed += 1
        return self._fed in self._finals

    def Result(self):  # noqa: N802
        return self._finals[self._fed]

    def FinalResult(self):  # noqa: N802
        return self._still_forming

    def PartialResult(self):  # noqa: N802
        return json.dumps({"partial": self._partials.get(self._fed, "")})


def _ranked(*texts):
    return json.dumps({"alternatives": [{"text": text, "confidence": 1.0} for text in texts]})


def test_nothing_is_heard_until_the_recognizer_settles_an_utterance():
    listening = Listening(RULES, _ScriptedRecognizer(finals={2: _ranked("next")}), sample_rate=RATE)

    assert listening.feed(_block(), captured_at=10.5) is None
    assert listening.feed(_block(), captured_at=11.0) == Heard(
        Recognition(phrase="next", heard="next"), spoken_at=10.5, peak=LOUD,
        audio=_block() + _block(), candidates={"next": 0})


def test_an_utterance_is_dated_from_the_first_block_the_recognizer_had_words_for():
    recognizer = _ScriptedRecognizer(finals={3: _ranked("left next")}, partials={2: "left"})
    listening = Listening(RULES, recognizer, sample_rate=RATE)
    listening.feed(_block(), captured_at=10.5)
    listening.feed(_block(), captured_at=11.0)

    heard = listening.feed(_block(), captured_at=11.5)

    assert heard.spoken_at == SECOND_BLOCK_BEGAN


def _scored(text, conf):
    return json.dumps({"text": text, "result": [{"conf": conf, "word": w} for w in text.split()]})


def test_a_miss_carries_the_unrestricted_reading_whenever_that_recognizer_finished_it():
    unrestricted = _ScriptedRecognizer(finals={1: _scored("what is next", 0.4)})
    listening = Listening(RULES, _ScriptedRecognizer(finals={2: _ranked("left net")}),
                          unrestricted=unrestricted, sample_rate=RATE)
    listening.feed(_block(), captured_at=10.5)

    heard = listening.feed(_block(), captured_at=11.0)

    assert heard.recognition == Recognition(
        unrecognized_text="left net", heard="left net", free_text="what is next")


def test_an_unrestricted_reading_still_forming_is_cut_short_when_the_grammar_settles_first():
    unrestricted = _ScriptedRecognizer(still_forming=_scored("what is", 0.4))
    listening = Listening(RULES, _ScriptedRecognizer(finals={1: _ranked("left net")}),
                          unrestricted=unrestricted, sample_rate=RATE)

    heard = listening.feed(_block(), captured_at=10.5)

    assert heard.recognition.free_text == "what is"


def test_the_words_still_forming_are_passed_on_each_time_they_change_and_cleared_at_the_end():
    forming = []
    recognizer = _ScriptedRecognizer(finals={4: _ranked("left next")},
                                     partials={1: "left", 2: "left", 3: "left next"})
    listening = Listening(RULES, recognizer, sample_rate=RATE, on_partial=forming.append)

    for number in range(4):
        listening.feed(_block(), captured_at=10.5 + number / 2)

    assert forming == ["left", "left next", ""]


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
