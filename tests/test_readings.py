from __future__ import annotations

import json

from voice_core.readings import Hypothesis, build_grammar, hypotheses, partial_text


def test_the_grammar_is_the_phrases_in_order_and_somewhere_for_everything_else_to_land():
    assert json.loads(build_grammar(["next", "left next", "back"])) == [
        "back", "left next", "next", "[unk]"]


def test_the_words_still_forming_are_what_a_partial_carries():
    assert partial_text(json.dumps({"partial": " left next "})) == "left next"


def test_a_partial_with_nothing_on_script_in_it_is_blank():
    assert partial_text(json.dumps({"partial": "[unk]"})) == ""
    assert partial_text(json.dumps({})) == ""
    assert partial_text("not json") == ""


def _scored(text, conf):
    words = [{"conf": conf, "word": word, "start": 0.0, "end": 0.1} for word in text.split()]
    return json.dumps({"text": text, "result": words})


def _ranked(*texts):
    return json.dumps({"alternatives": [
        {"text": text, "confidence": 100.0 - rank,
         "result": [{"word": word, "start": 0.0, "end": 0.1} for word in text.split()]}
        for rank, text in enumerate(texts)]})


def test_a_single_reading_comes_with_each_words_score():
    assert hypotheses(_scored("left next", 0.9)) == [Hypothesis("left next", (0.9, 0.9))]


def test_ranked_readings_come_best_first_and_unscored():
    assert hypotheses(_ranked("left net", "left next")) == [
        Hypothesis("left net"), Hypothesis("left next")]


def test_each_word_of_a_reading_comes_with_when_it_was_said():
    raw = json.dumps({"alternatives": [{"text": "left next", "confidence": 1.0, "result": [
        {"word": "left", "start": 1.5, "end": 1.8}, {"word": "next", "start": 1.8, "end": 2.2}]}]})

    [reading] = hypotheses(raw)

    assert reading.times == ((1.5, 1.8), (1.8, 2.2))


def test_a_reading_whose_words_carry_no_times_has_none():
    [reading] = hypotheses(json.dumps({"alternatives": [{"text": "left next", "confidence": 1.0}]}))

    assert reading.times == ()


def test_a_reading_with_no_words_in_it_is_no_reading():
    assert hypotheses(_ranked("", "next")) == [Hypothesis("next")]
    assert hypotheses(json.dumps({"text": "  "})) == []
    assert hypotheses("") == []
