from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from voice_core.readings import UNKNOWN, Hypothesis, hypotheses

# vosk reads words out of the digital silence his microphone's noise suppressor cuts to
# between utterances; speech clears this.
SILENT_UTTERANCE_PEAK = 300


@dataclass(frozen=True)
class CommandRules:
    phrases: frozenset[str]
    never_repaired: Callable[[str], bool]
    confidence_threshold: float = 0.7
    silent_peak: int = SILENT_UTTERANCE_PEAK
    stands_alone: Callable[[str], bool] | None = None
    written: Callable[[str], str | None] | None = None
    said_in: Callable[[str], str | None] | None = None

    def other_languages(self) -> frozenset[str]:
        return frozenset(filter(None, map(self.said_in, self.phrases))) if self.said_in else frozenset()


@dataclass(frozen=True)
class Recognition:
    phrase: str | None = None
    rank: int = 0
    refused_phrase: str | None = None
    unconfirmed_phrase: str | None = None
    unrecognized_text: str | None = None
    heard: str | None = None
    silent_reading: str | None = None
    free_text: str | None = None


def interpret(grammar_json: str, free_json: str, *, rules: CommandRules, peak: int) -> Recognition:
    readings = _as_said(hypotheses(grammar_json), rules)
    spoken = next((reading.text for reading in readings if reading.text not in ("", UNKNOWN)), None)
    if peak < rules.silent_peak:
        return Recognition(silent_reading=spoken)
    free = next(iter(_as_said(hypotheses(free_json), rules)), None)
    free_text = free.text if free and free.text != UNKNOWN else None
    for rank, hypothesis in enumerate(readings):
        if hypothesis.text not in rules.phrases:
            continue
        if hypothesis.confidences and not _clears(hypothesis.confidences, rules.confidence_threshold):
            return Recognition(refused_phrase=hypothesis.text, heard=spoken, free_text=free_text)
        if rank and (rules.never_repaired(hypothesis.text)
                     or not _shares_a_word(hypothesis.text, spoken)):
            continue
        return Recognition(phrase=hypothesis.text, rank=rank, heard=spoken)
    if spoken:
        return Recognition(unrecognized_text=spoken, heard=spoken, free_text=free_text)
    if free_text and free.confidences and _clears(free.confidences, rules.confidence_threshold):
        return Recognition(unrecognized_text=free_text)
    return Recognition()


# vosk reads the click his microphone's noise suppressor opens with as "the".
_THE_MICROPHONE_OPENING = "the"


def _as_said(readings: Sequence[Hypothesis], rules: CommandRules) -> list[Hypothesis]:
    return [reading if reading.text in rules.phrases else _without_the_opening(reading)
            for reading in readings]


def _without_the_opening(reading: Hypothesis) -> Hypothesis:
    words = reading.text.split()
    first, last = 0, len(words)
    while first < last and words[first] == _THE_MICROPHONE_OPENING:
        first += 1
    while last > first and words[last - 1] == _THE_MICROPHONE_OPENING:
        last -= 1
    return replace(reading, text=" ".join(words[first:last]), times=reading.times[first:last])


def where_said(grammar_json: str, phrases: Iterable[str], *,
               rules: CommandRules) -> tuple[float, float] | None:
    wanted = set(phrases)
    spans = [(reading.times[0][0], reading.times[-1][1])
             for reading in _as_said(hypotheses(grammar_json), rules)
             if reading.text in wanted and reading.times]
    if not spans:
        return None
    return min(start for start, _end in spans), max(end for _start, end in spans)


def _shares_a_word(reading: str, first_choice: str) -> bool:
    return bool(set(reading.split()) & set(first_choice.split()))


def _clears(confidences: Sequence[float], threshold: float) -> bool:
    return math.fsum(confidences) >= threshold * len(confidences)


def candidates(grammar_json: str, *, rules: CommandRules, peak: int) -> dict[str, int]:
    if peak < rules.silent_peak:
        return {}
    found: dict[str, int] = {}
    for rank, reading in enumerate(_as_said(hypotheses(grammar_json), rules)):
        if reading.text in rules.phrases and not (rank and rules.never_repaired(reading.text)):
            found.setdefault(reading.text, rank)
    return found
