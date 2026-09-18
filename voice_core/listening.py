from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from voice_core.capture import CaptureLevel, Utterance
from voice_core.commands import CommandRules, Recognition, candidates, interpret
from voice_core.readings import partial_text


@dataclass(frozen=True)
class Heard:
    recognition: Recognition
    spoken_at: float
    peak: int
    audio: bytes
    # Every phrase among the ranked readings, with its rank: what a second engine may choose from.
    candidates: Mapping[str, int]


class Listening:
    def __init__(self, rules: CommandRules, recognizer, *, sample_rate: int,
                 unrestricted=None, on_partial: Callable[[str], None] | None = None) -> None:
        self._on_partial = on_partial
        self._partial = ""
        self._rules = rules
        self._recognizer = recognizer
        self._unrestricted = unrestricted
        # The unrestricted recognizer ends its utterances on its own schedule,
        # so its latest reading is banked until the grammar one settles.
        self._banked = ""
        self._sample_rate = sample_rate
        self._level = CaptureLevel()
        self._utterance = Utterance()

    def feed(self, data: bytes, *, captured_at: float) -> Heard | None:
        self._level.note_block(data)
        block_started_at = captured_at - (len(data) / 2) / self._sample_rate
        settled = self._recognizer.AcceptWaveform(data)
        if self._unrestricted is not None and self._unrestricted.AcceptWaveform(data):
            self._banked = self._unrestricted.Result()
        if not settled:
            forming = partial_text(self._recognizer.PartialResult())
            self._note_partial(forming)
            self._utterance.note_block(
                data, block_started_at=block_started_at, has_partial=bool(forming))
            return None
        self._note_partial("")
        peak = self._level.take_utterance()
        ranked = self._recognizer.Result()
        recognition = interpret(ranked, self._take_unrestricted(), rules=self._rules, peak=peak)
        spoken_at, audio = self._utterance.take(final_block=data, fallback=block_started_at)
        return Heard(recognition, spoken_at=spoken_at, peak=peak, audio=audio,
                     candidates=candidates(ranked, rules=self._rules, peak=peak))

    def take_recent_level(self) -> int:
        return self._level.take_recent()

    def _take_unrestricted(self) -> str:
        banked, self._banked = self._banked, ""
        if banked or self._unrestricted is None:
            return banked
        return self._unrestricted.FinalResult()

    def _note_partial(self, forming: str) -> None:
        if forming != self._partial:
            self._partial = forming
            if self._on_partial is not None:
                self._on_partial(forming)


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
