from __future__ import annotations

from collections.abc import Sequence

_UNITS = {word: value for value, word in enumerate([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen"])}
_TENS = {word: value for value, word in zip(
    range(20, 100, 10),
    ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], strict=True)}
_LAST_SINGLE_DIGIT = 9


def numbers_as_digits(words: Sequence[str]) -> list[str]:
    out: list[str] = []
    tens = None
    for word in words:
        if tens is not None and 0 < _UNITS.get(word, 0) <= _LAST_SINGLE_DIGIT:
            out[-1] = str(tens + _UNITS[word])
            tens = None
            continue
        tens = _TENS.get(word)
        if word == "hundred" and out and out[-1].isdigit():
            out[-1] = str(int(out[-1]) * 100)
        elif word in _UNITS or word in _TENS:
            out.append(str(_UNITS.get(word, tens)))
        else:
            out.append(word)
    return out
