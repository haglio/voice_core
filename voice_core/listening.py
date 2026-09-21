from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from voice_core.capture import CaptureLevel
from voice_core.commands import CommandRules, Recognition, candidates, interpret
from voice_core.pauses import PauseSegmenter
from voice_core.readings import UNKNOWN, hypotheses, joined, partial_text


@dataclass(frozen=True)
class Heard:
    recognition: Recognition
    spoken_at: float
    peak: int
    audio: bytes
    # Every phrase among the ranked readings, with its rank: what a second engine may choose from.
    candidates: Mapping[str, int]


@dataclass(frozen=True)
class Recognizers:
    grammar: Any
    # Reads the same audio with no phrase list, only to caption what the grammar one missed.
    unrestricted: Any = None


class Listening:
    def __init__(self, rules: CommandRules, recognizers: Recognizers, segmenter: PauseSegmenter, *,
                 on_partial: Callable[[str], None] | None = None,
                 for_dictation: bool = False) -> None:
        self._rules = rules
        self._recognizer = recognizers.grammar
        self._unrestricted = recognizers.unrestricted
        self._segmenter = segmenter
        self._on_partial = on_partial
        self._for_dictation = for_dictation
        self._partial = ""
        self._level = CaptureLevel()
        self._began_at = 0.0
        self._settled_partway: list[str] = []

    def feed(self, data: bytes, *, started_at: float) -> Heard | None:
        spoke_before = self._segmenter.speaking
        utterance = self._segmenter.push(data)
        if self._segmenter.speaking and not spoke_before:
            self._begin_utterance(at=started_at)
        self._level.note_block(data)
        if self._recognizer.AcceptWaveform(data):
            self._settled_partway.append(self._recognizer.Result())
        if utterance is not None:
            self._note_partial("")
            return self._heard(utterance)
        if self._segmenter.speaking:
            self._note_partial(partial_text(self._recognizer.PartialResult()))
        return None

    def take_recent_level(self) -> int:
        return self._level.take_recent()

    def _begin_utterance(self, *, at: float) -> None:
        self._began_at = at
        self._recognizer.Reset()
        self._level.take_utterance()
        self._settled_partway = []

    def _heard(self, audio: bytes) -> Heard:
        peak = self._level.take_utterance()
        readings = [*self._settled_partway, self._recognizer.FinalResult()]
        said = [ranked for ranked in readings if _first_choice(ranked)] or readings[-1:]
        meant = self._reading_meant(said, peak=peak)
        if meant is None:
            talk = " ".join(map(_first_choice, said))
            return Heard(Recognition(unrecognized_text=talk, heard=talk),
                         spoken_at=self._began_at, peak=peak, audio=audio, candidates={})
        recognition = interpret(meant, "", rules=self._rules, peak=peak)
        if not recognition.phrase and self._unrestricted is not None:
            recognition = interpret(meant, self._caption(audio), rules=self._rules, peak=peak)
        return Heard(recognition, spoken_at=self._began_at, peak=peak, audio=audio,
                     candidates=candidates(meant, rules=self._rules, peak=peak))

    def _reading_meant(self, said: list[str], *, peak: int) -> str | None:
        if len(said) == 1:
            return said[0]
        if self._for_dictation:
            return None
        return next((ranked for ranked in said
                     if interpret(ranked, "", rules=self._rules, peak=peak).phrase), None)

    def _caption(self, audio: bytes) -> str:
        settled = self._unrestricted.Result() if self._unrestricted.AcceptWaveform(audio) else ""
        return joined([settled, self._unrestricted.FinalResult()])

    def _note_partial(self, forming: str) -> None:
        if forming != self._partial:
            self._partial = forming
            if self._on_partial is not None:
                self._on_partial(forming)


def _first_choice(ranked: str) -> str:
    readings = hypotheses(ranked)
    return readings[0].text if readings and readings[0].text != UNKNOWN else ""


def outcome_line(heard: Heard, *, now: float, rules: CommandRules) -> tuple[int, str]:
    recognition, peak = heard.recognition, heard.peak
    unrestricted = recognition.free_text or ""
    if recognition.phrase:
        repaired = (f" -- the recognizer's choice {recognition.rank + 1}, under "
                    f"{recognition.heard!r}, which is no command" if recognition.rank else "")
        return logging.INFO, (
            f"Voice command: {recognition.phrase!r}{repaired} "
            f"(spoken {now - heard.spoken_at:.2f}s before recognition, peak {peak})")
    if recognition.unconfirmed_phrase:
        return logging.INFO, (f"Voice: heard {recognition.unconfirmed_phrase!r} but the second "
                              f"engine read {unrestricted!r} (peak {peak})")
    if recognition.refused_phrase:
        return logging.INFO, (
            f"Voice: heard {recognition.refused_phrase!r} but its confidence was under "
            f"{rules.confidence_threshold:.2f} (unrestricted reading {unrestricted!r}, peak {peak})")
    if recognition.unrecognized_text:
        return logging.INFO, (f"Unrecognized speech: {recognition.unrecognized_text!r} "
                              f"(unrestricted reading {unrestricted!r}, peak {peak})")
    if recognition.silent_reading:
        return logging.INFO, (
            f"Voice: ignored {recognition.silent_reading!r} read from silence (peak {peak})")
    return logging.DEBUG, f"Voice: an utterance ended with nothing in it (peak {peak})"
