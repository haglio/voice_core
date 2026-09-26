from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

from voice_core.commands import CommandRules, Recognition
from voice_core.listening import Heard
from voice_core.second_opinion import chosen_among, settle


def test_a_reading_that_is_one_of_the_candidates_chooses_it_whatever_its_case_and_punctuation():
    assert chosen_among("Left next.", ("next", "left next")) == "left next"


def test_a_reading_that_is_none_of_them_chooses_nothing():
    assert chosen_among("what is next", ("next", "left next")) is None
    assert chosen_among("", ("next",)) is None


def test_a_phrase_the_engine_repeated_is_still_that_phrase():
    assert chosen_among("left next, left next, left next.", ("left next",)) == "left next"


def test_the_word_a_microphone_click_became_at_either_end_of_a_reading_is_no_part_of_it():
    assert chosen_among("And enter VR.", ("enter vr",)) == "enter vr"
    assert chosen_among("The next, uh", ("next",)) == "next"
    assert chosen_among("Go next.", ("next",)) is None


def test_a_number_written_in_digits_is_the_number_said_in_words():
    assert chosen_among("Amp 50", ("amp fifty", "amp fifteen")) == "amp fifty"
    assert chosen_among("clip seconds 25", ("clip seconds twenty five",)) == "clip seconds twenty five"
    assert chosen_among("amp 100", ("amp one hundred",)) == "amp one hundred"


def test_letters_the_engine_wrote_joined_up_are_the_letters_said_one_by_one():
    assert chosen_among("OSR2 off.", ("o s r two off",)) == "o s r two off"
    assert chosen_among("Exit VR", ("exit v r", "exit vr")) == "exit v r"


def test_a_phrase_spelled_by_sound_for_one_engine_is_matched_as_the_app_writes_it():
    written = {"go now mode": "genau mode"}

    assert chosen_among("Genau mode.", ("go now mode",), written=written.get) == "go now mode"
    assert chosen_among("go now mode", ("go now mode",), written=written.get) == "go now mode"


RULES = CommandRules(
    phrases=frozenset({"next", "lock", "left next", "quit"}),
    never_rescued=frozenset({"quit"}).__contains__,
    stands_alone=lambda phrase: " " in phrase,
)
AUDIO = b"\x01\x00" * 8


def _heard(recognition, candidates):
    return Heard(recognition, spoken_at=1.0, peak=2000, audio=AUDIO, candidates=candidates)


_never_asked = Mock(side_effect=AssertionError("the second engine was asked"))


def test_a_phrase_the_app_lets_stand_alone_is_not_put_to_the_second_engine():
    heard = _heard(Recognition(phrase="left next", heard="left next"), {"left next": 0})

    assert settle(heard, rules=RULES, read=_never_asked) == heard


def test_a_first_choice_the_second_engine_also_reads_is_the_command():
    heard = _heard(Recognition(phrase="next", heard="next"), {"next": 0, "lock": 2})
    asked = []

    settled = settle(heard, rules=RULES, read=lambda audio, hint: asked.append((audio, hint)) or "Next.")

    assert settled == heard
    assert asked == [(AUDIO, "next, lock")]


def test_the_second_engine_reads_only_the_stretch_the_first_heard_the_phrase_in():
    heard = replace(_heard(Recognition(phrase="next", heard="next"), {"next": 0}),
                    phrase_audio=AUDIO[:4])
    asked = []

    settle(heard, rules=RULES, read=lambda audio, hint: asked.append(audio) or "Next.")

    assert asked == [AUDIO[:4]]


def test_a_phrase_the_second_engine_could_not_read_as_recorded_is_listened_to_again_closer():
    heard = _heard(Recognition(phrase="next", heard="next"), {"next": 0})

    settled = settle(heard, rules=RULES, read=lambda audio, hint: "",
                     read_closely=lambda audio, hint: "Next.")

    assert settled == heard


def test_a_first_choice_the_second_engine_reads_otherwise_is_not_acted_on():
    heard = _heard(Recognition(phrase="next", heard="next"), {"next": 0})

    settled = settle(heard, rules=RULES, read=lambda audio, hint: "and then we went")

    assert settled.recognition == Recognition(
        unconfirmed_phrase="next", heard="next", free_text="and then we went")


def test_what_the_first_engine_could_not_settle_the_second_may_choose_among_its_readings():
    heard = _heard(Recognition(unrecognized_text="eyes", heard="eyes"), {"lock": 1, "next": 3})

    settled = settle(heard, rules=RULES, read=lambda audio, hint: "next")

    assert settled.recognition == Recognition(phrase="next", rank=3, heard="eyes")


def test_a_miss_the_second_engine_cannot_place_either_stays_the_miss_it_was_with_its_reading():
    heard = _heard(Recognition(unrecognized_text="eyes", heard="eyes"), {"lock": 1})

    settled = settle(heard, rules=RULES, read=lambda audio, hint: "ice")

    assert settled.recognition == Recognition(unrecognized_text="eyes", heard="eyes", free_text="ice")


def test_with_no_candidates_nobody_is_asked():
    heard = _heard(Recognition(unrecognized_text="left net", heard="left net"), {})

    assert settle(heard, rules=RULES, read=_never_asked) == heard


def test_where_the_engines_disagree_the_second_ones_choice_among_the_readings_wins():
    heard = _heard(Recognition(phrase="next", heard="next"), {"next": 0, "lock": 2})

    settled = settle(heard, rules=RULES, read=lambda audio, hint: "Lock.")

    assert settled.recognition == Recognition(phrase="lock", rank=2, heard="next")


def test_the_second_engine_is_shown_each_phrase_as_the_app_writes_it():
    rules = CommandRules(phrases=frozenset({"go now"}), never_rescued=RULES.never_rescued,
                         written={"go now": "genau"}.get)
    heard = _heard(Recognition(phrase="go now", heard="go now"), {"go now": 0})
    hints = []

    settled = settle(heard, rules=rules, read=lambda audio, hint: hints.append(hint) or "Genau.")

    assert (hints, settled.recognition.phrase) == (["genau"], "go now")
