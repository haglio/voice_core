from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from voice_core.readings import UNKNOWN, hypotheses

# The noise suppressor on his microphone cuts to digital silence between
# utterances, and the recognizer reads words out of it; speech clears this.
SILENT_UTTERANCE_PEAK = 300


@dataclass(frozen=True)
class CommandRules:
    phrases: frozenset[str]
    never_rescued: Callable[[str], bool]
    confidence_threshold: float = 0.7
    silent_peak: int = SILENT_UTTERANCE_PEAK


@dataclass(frozen=True)
class Recognition:
    phrase: str | None = None
    # 0 the recognizer's first choice, higher a repair taken from under it.
    rank: int = 0
    refused_phrase: str | None = None
    unrecognized_text: str | None = None
    heard: str | None = None
    silent_reading: str | None = None
    free_text: str | None = None


def interpret(grammar_json: str, free_json: str, *, rules: CommandRules, peak: int) -> Recognition:
    readings = hypotheses(grammar_json)
    spoken = next((reading.text for reading in readings if reading.text != UNKNOWN), None)
    if peak < rules.silent_peak:
        return Recognition(silent_reading=spoken)
    free = next(iter(hypotheses(free_json)), None)
    free_text = free.text if free and free.text != UNKNOWN else None
    for rank, hypothesis in enumerate(readings):
        if hypothesis.text not in rules.phrases:
            continue
        if hypothesis.confidences and not _clears(hypothesis.confidences, rules.confidence_threshold):
            # A lower-ranked reading is less likely still, so this ends the search.
            return Recognition(refused_phrase=hypothesis.text, heard=spoken, free_text=free_text)
        if rank and (rules.never_rescued(hypothesis.text)
                     or not _shares_a_word(hypothesis.text, spoken)):
            continue
        return Recognition(phrase=hypothesis.text, rank=rank, heard=spoken)
    if spoken:
        return Recognition(unrecognized_text=spoken, heard=spoken, free_text=free_text)
    if free_text and free.confidences and _clears(free.confidences, rules.confidence_threshold):
        return Recognition(unrecognized_text=free_text)
    return Recognition()


def _shares_a_word(reading: str, first_choice: str) -> bool:
    return bool(set(reading.split()) & set(first_choice.split()))


def _clears(confidences: Sequence[float], threshold: float) -> bool:
    # As a sum against the bar times the count: the quotient rounds three words
    # spoken exactly at the bar under it.
    return math.fsum(confidences) >= threshold * len(confidences)
