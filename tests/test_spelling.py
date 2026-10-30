from __future__ import annotations

from voice_core.spelling import numbers_as_digits


def test_a_number_said_in_words_is_written_in_digits():
    assert numbers_as_digits(["amp", "fifty"]) == ["amp", "50"]
    assert numbers_as_digits(["clip", "seconds", "twenty", "five"]) == ["clip", "seconds", "25"]
    assert numbers_as_digits(["amp", "one", "hundred"]) == ["amp", "100"]
    assert numbers_as_digits(["center", "zero"]) == ["center", "0"]


def test_a_number_already_in_digits_and_every_word_that_is_no_number_stay_as_they_were():
    assert numbers_as_digits(["amp", "50"]) == ["amp", "50"]
    assert numbers_as_digits(["left", "next"]) == ["left", "next"]
    assert numbers_as_digits(["twenty", "next"]) == ["20", "next"]
