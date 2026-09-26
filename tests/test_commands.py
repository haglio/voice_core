from __future__ import annotations

import json

from voice_core.commands import (
    SILENT_UTTERANCE_PEAK,
    CommandRules,
    Recognition,
    candidates,
    interpret,
    where_said,
)

SPOKEN = 2000
RULES = CommandRules(
    phrases=frozenset({"left next", "left lock", "next", "skip", "quit", "help", "amp up",
                       "park it", "main reset", "main video mode"}),
    never_rescued=frozenset({"quit", "park it", "main reset"}).__contains__,
)


def _scored(text, conf):
    words = [{"conf": conf, "word": word, "start": 0.0, "end": 0.1} for word in text.split()]
    return json.dumps({"text": text, "result": words})


def _ranked(*texts):
    return json.dumps({"alternatives": [
        {"text": text, "confidence": 100.0 - rank,
         "result": [{"word": word, "start": 0.0, "end": 0.1} for word in text.split()]}
        for rank, text in enumerate(texts)]})


def test_a_first_choice_that_is_a_phrase_is_the_command():
    heard = interpret(_ranked("left next"), "", rules=RULES, peak=SPOKEN)

    assert heard == Recognition(phrase="left next", heard="left next")


def test_a_near_miss_is_repaired_from_the_readings_ranked_under_it():
    heard = interpret(_ranked("left net", "left next"), "", rules=RULES, peak=SPOKEN)

    assert heard == Recognition(phrase="left next", rank=1, heard="left net")


def test_a_repair_must_share_a_word_with_the_first_choice():
    unrelated = interpret(_ranked("half", "help"), "", rules=RULES, peak=SPOKEN)
    related = interpret(_ranked("up", "amp up"), "", rules=RULES, peak=SPOKEN)

    assert unrelated == Recognition(unrecognized_text="half", heard="half")
    assert related == Recognition(phrase="amp up", rank=1, heard="up")


def test_a_repair_never_lands_on_a_phrase_the_app_rules_out():
    for first, under_it in (("it", "park it"), ("main", "main reset"), ("net", "quit")):
        repaired = interpret(_ranked(first, under_it), "", rules=RULES, peak=SPOKEN)
        first_choice = interpret(_ranked(under_it), "", rules=RULES, peak=SPOKEN)

        assert repaired == Recognition(unrecognized_text=first, heard=first), under_it
        assert first_choice.phrase == under_it


def test_what_lands_outside_the_phrases_is_never_the_reading_reported():
    heard = interpret(_ranked("[unk]", "left net"), "", rules=RULES, peak=SPOKEN)

    assert heard == Recognition(unrecognized_text="left net", heard="left net")


def test_a_phrase_ranked_under_nothing_on_script_is_still_a_repair():
    ruled_out = interpret(_ranked("[unk]", "quit"), "", rules=RULES, peak=SPOKEN)
    allowed = interpret(_ranked("[unk]", "next"), "", rules=RULES, peak=SPOKEN)

    assert ruled_out == Recognition(unrecognized_text="quit", heard="quit")
    assert allowed == Recognition(phrase="next", rank=1, heard="next")


def test_a_scored_phrase_under_the_bar_is_refused_by_name():
    heard = interpret(_scored("skip", 0.3), "", rules=RULES, peak=SPOKEN)

    assert heard == Recognition(refused_phrase="skip", heard="skip")


def test_three_words_scored_exactly_at_the_bar_clear_it():
    heard = interpret(_scored("main video mode", 0.7), "", rules=RULES, peak=SPOKEN)

    assert heard == Recognition(phrase="main video mode", heard="main video mode")


def test_a_miss_carries_what_the_unrestricted_recognizer_made_of_it():
    unmatched = interpret(_ranked("left net"), _scored("what is next", 0.3), rules=RULES, peak=SPOKEN)
    refused = interpret(_scored("skip", 0.3), _scored("skip it", 0.3), rules=RULES, peak=SPOKEN)
    nothing_free = interpret(_ranked("left net"), json.dumps({"text": "[unk]"}), rules=RULES,
                             peak=SPOKEN)

    assert unmatched == Recognition(unrecognized_text="left net", heard="left net",
                                    free_text="what is next")
    assert refused == Recognition(refused_phrase="skip", heard="skip", free_text="skip it")
    assert nothing_free.free_text is None


def test_with_nothing_on_script_a_confident_unrestricted_reading_is_what_was_heard():
    confident = interpret(json.dumps({"text": "[unk]"}), _scored("skip it now", 0.7),
                          rules=RULES, peak=SPOKEN)
    mumbled = interpret(json.dumps({"text": "[unk]"}), _scored("mumble", 0.3),
                        rules=RULES, peak=SPOKEN)

    assert confident == Recognition(unrecognized_text="skip it now")
    assert mumbled == Recognition()


def test_a_reading_out_of_silence_is_set_aside_and_never_repaired():
    heard = interpret(_ranked("half", "help"), "", rules=RULES, peak=SILENT_UTTERANCE_PEAK - 1)

    assert heard == Recognition(silent_reading="half")


def test_a_quiet_utterance_that_reaches_the_floor_still_counts():
    heard = interpret(_ranked("skip"), "", rules=RULES, peak=SILENT_UTTERANCE_PEAK)

    assert heard.phrase == "skip"


def test_the_candidates_are_every_ranked_reading_that_is_a_phrase_with_the_rank_it_first_had():
    found = candidates(_ranked("left net", "left next", "[unk]", "next", "left next"), rules=RULES,
                       peak=SPOKEN)

    assert found == {"left next": 1, "next": 3}
    assert list(found) == ["left next", "next"]


def test_silence_has_no_candidates():
    assert candidates(_ranked("next"), rules=RULES, peak=SILENT_UTTERANCE_PEAK - 1) == {}


def test_a_phrase_ruled_out_of_repairs_is_a_candidate_only_as_the_first_choice():
    assert candidates(_ranked("net", "quit", "next"), rules=RULES, peak=SPOKEN) == {"next": 2}
    assert candidates(_ranked("quit", "next"), rules=RULES, peak=SPOKEN) == {"quit": 0, "next": 1}


def test_a_the_at_either_end_of_a_reading_is_the_microphone_opening_not_a_word_he_said():
    led = interpret(_ranked("the left next"), "", rules=RULES, peak=SPOKEN)
    trailed = interpret(_ranked("left next the"), "", rules=RULES, peak=SPOKEN)

    assert led == trailed == Recognition(phrase="left next", heard="left next")


def test_a_phrase_with_the_microphone_opening_on_it_is_a_candidate_as_the_phrase_itself():
    found = candidates(_ranked("the left net", "the left next", "next the"), rules=RULES, peak=SPOKEN)

    assert found == {"left next": 1, "next": 2}


def test_the_microphone_opening_on_its_own_is_nothing_said_at_all():
    heard = interpret(_ranked("the", "the the"), _scored("the", 0.9), rules=RULES, peak=SPOKEN)

    assert heard == Recognition()


def test_a_reading_that_is_no_phrase_is_reported_without_the_microphone_opening_on_it():
    heard = interpret(_ranked("the left net"), "", rules=RULES, peak=SPOKEN)

    assert heard == Recognition(unrecognized_text="left net", heard="left net")


def _timed(*readings):
    return json.dumps({"alternatives": [
        {"text": " ".join(word for word, *_times in words), "confidence": 100.0 - rank,
         "result": [{"word": word, "start": start, "end": end} for word, start, end in words]}
        for rank, words in enumerate(readings)]})


def test_a_phrase_was_said_from_its_first_word_to_its_last_leaving_the_microphone_opening_out():
    raw = _timed([("the", 0.2, 9.0), ("left", 9.1, 9.4), ("next", 9.4, 9.8)])

    assert where_said(raw, ["left next"], rules=RULES) == (9.1, 9.8)


def test_several_phrases_were_said_across_the_stretch_that_holds_them_all():
    raw = _timed([("next", 1.0, 1.3)], [("left", 0.7, 1.0), ("next", 1.0, 1.3)],
                 [("skip", 2.0, 2.4)])

    assert where_said(raw, ["next", "left next"], rules=RULES) == (0.7, 1.3)


def test_where_a_phrase_was_said_is_unknown_when_its_words_carry_no_times():
    untimed = json.dumps({"alternatives": [{"text": "left next", "confidence": 1.0}]})

    assert where_said(untimed, ["left next"], rules=RULES) is None


def test_what_was_heard_is_the_first_reading_with_words_left_in_it():
    heard = interpret(_ranked("the", "left net", "left next"), "", rules=RULES, peak=SPOKEN)

    assert heard == Recognition(phrase="left next", rank=2, heard="left net")
