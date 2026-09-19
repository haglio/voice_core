from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from voice_core.capture import KEPT_BLOCKS, CaptureLevel, Utterance
from voice_core.commands import CommandRules, Recognition, candidates, interpret
from voice_core.pauses import PauseSegmenter
from voice_core.readings import UNKNOWN, hypotheses, partial_text


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
    def __init__(self, rules: CommandRules, recognizers: Recognizers, *, sample_rate: int,
                 on_partial: Callable[[str], None] | None = None,
                 kept_blocks: int = KEPT_BLOCKS) -> None:
        self._on_partial = on_partial
        self._partial = ""
        self._rules = rules
        self._recognizer = recognizers.grammar
        self._unrestricted = recognizers.unrestricted
        # The unrestricted recognizer ends its utterances on its own schedule,
        # so its latest reading is banked until the grammar one settles.
        self._banked = ""
        self._sample_rate = sample_rate
        self._level = CaptureLevel()
        self._utterance = Utterance(kept_blocks=kept_blocks)

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


class PausedListening:
    """Listening for an app that also takes dictation: an utterance ends where the speaker
    pauses (see pauses.PauseSegmenter), and the recognizer is made to finish there."""

    def __init__(self, rules: CommandRules, recognizers: Recognizers, segmenter: PauseSegmenter, *,
                 sample_rate: int) -> None:
        self._rules = rules
        self._recognizer = recognizers.grammar
        self._segmenter = segmenter
        self._sample_rate = sample_rate
        self._level = CaptureLevel()
        self._began_at: float | None = None
        self._settled_partway: list[str] = []

    def feed(self, data: bytes, *, captured_at: float) -> Heard | None:
        self._level.note_block(data)
        settled = self._recognizer.Result() if self._recognizer.AcceptWaveform(data) else None
        was_speaking = self._segmenter.speaking
        audio = self._segmenter.push(data)
        if was_speaking and audio is None and not self._segmenter.speaking:
            self._forget_a_sound_too_short_to_be_a_word()
            return None
        if settled is not None and (self._segmenter.speaking or audio is not None):
            self._settled_partway.append(settled)
        if self._began_at is None and self._segmenter.speaking:
            self._began_at = captured_at - (len(data) / 2) / self._sample_rate
        if audio is None:
            return None
        peak = self._level.take_utterance()
        ranked = self._recognizer.FinalResult()
        began_at, self._began_at = self._began_at, None
        partway, self._settled_partway = self._settled_partway, []
        if partway:
            # A command is one breath; this was several readings' worth of talk.
            said = " ".join(reading[0].text for reading in map(hypotheses, [*partway, ranked])
                            if reading and reading[0].text != UNKNOWN)
            return Heard(Recognition(unrecognized_text=said or None, heard=said or None),
                         spoken_at=began_at, peak=peak, audio=audio, candidates={})
        return Heard(interpret(ranked, "", rules=self._rules, peak=peak), spoken_at=began_at,
                     peak=peak, audio=audio,
                     candidates=candidates(ranked, rules=self._rules, peak=peak))

    def take_recent_level(self) -> int:
        return self._level.take_recent()

    def _forget_a_sound_too_short_to_be_a_word(self) -> None:
        self._recognizer.FinalResult()
        self._level.take_utterance()
        self._began_at = None
        self._settled_partway = []
