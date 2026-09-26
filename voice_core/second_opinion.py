from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace

from voice_core.commands import CommandRules, Recognition
from voice_core.listening import Heard

_UNITS = {word: value for value, word in enumerate([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen"])}
_TENS = {word: value for value, word in zip(
    range(20, 100, 10),
    ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], strict=True)}
_LAST_SINGLE_DIGIT = 9


def _numbers_as_digits(words: Sequence[str]) -> list[str]:
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


def _squashed(text: str) -> str:
    return "".join(_numbers_as_digits(re.findall(r"[a-z]+|\d+", text.lower())))


def chosen_among(reading: str, candidates: Sequence[str], *,
                 written: Callable[[str], str | None] | None = None) -> str | None:
    heard = _squashed(reading)
    for phrase in candidates:
        spellings = (phrase, written(phrase) if written else None)
        if any(_is_repeated(heard, _squashed(spelling)) for spelling in spellings if spelling):
            return phrase
    return None


def _is_repeated(heard: str, said: str) -> bool:
    # Handed a hint, whisper often writes the phrase it heard several times over.
    return bool(heard and said) and heard == said * (len(heard) // len(said))


def settle(heard: Heard, *, rules: CommandRules, read: Callable[[bytes, str], str],
           read_closely: Callable[[bytes, str], str] | None = None) -> Heard:
    first = heard.recognition
    stands_alone = bool(first.phrase and rules.stands_alone and rules.stands_alone(first.phrase))
    if stands_alone or not heard.candidates:
        return heard
    audio = heard.phrase_audio or heard.audio
    hint = ", ".join(
        (rules.written and rules.written(phrase)) or phrase for phrase in heard.candidates)
    reading = read(audio, hint)
    chosen = chosen_among(reading, tuple(heard.candidates), written=rules.written)
    if chosen is None and read_closely is not None:
        reading = read_closely(audio, hint)
        chosen = chosen_among(reading, tuple(heard.candidates), written=rules.written)
    if chosen is None:
        return replace(heard, recognition=replace(
            first, phrase=None, rank=0, unconfirmed_phrase=first.phrase, free_text=reading))
    return replace(heard, recognition=Recognition(
        phrase=chosen, rank=heard.candidates[chosen], heard=first.heard))
