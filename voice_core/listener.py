from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from voice_core.capture import AudioStall
from voice_core.commands import CommandRules
from voice_core.listening import Heard, Listening, outcome_line
from voice_core.microphone import probe_input_device, resolve_input_device
from voice_core.miss_clips import save_miss_audio
from voice_core.readings import build_grammar

logger = logging.getLogger(__name__)

BLOCK_SAMPLES = 8000

# Vosk's grammar restricts the vocabulary, not the phrases: it decodes any
# sequence of the phrases' words, so the right phrase often sits in the rankings
# under a sequence that is no command.
GRAMMAR_ALTERNATIVES = 5

# Without it, a session where every phrase missed and one where the microphone
# was dead leave the same log.
LISTEN_HEARTBEAT_S = 60.0


class RecognizerUnavailable(Exception):
    pass


class MicrophoneUnavailable(Exception):
    pass


def _reason(exc: Exception) -> str:
    return str(exc) or type(exc).__name__


def why_unavailable() -> str:
    # Absent, present without PortAudio, or broken at import: unavailable either
    # way, and the message is what the app has to show for it.
    try:
        importlib.import_module("sounddevice")
        importlib.import_module("vosk")
    except Exception as exc:
        return _reason(exc)
    return ""


@dataclass(frozen=True)
class ListenerSettings:
    model_name: str
    device_name: str | None = None
    sample_rate: int = 16000
    caption_misses: bool = False
    miss_dir: Path | None = None
    poll_seconds: float = 0.5


@dataclass(frozen=True)
class ListenerEvents:
    heard: Callable[[Heard], None]
    partial: Callable[[str], None] | None = None
    stalled: Callable[[float], None] | None = None
    recovered: Callable[[], None] | None = None
    # Whether the room is being listened to: what a muted app mishears is not kept.
    keeps_misses: Callable[[], bool] | None = None


@dataclass(frozen=True)
class Engines:
    vosk: Any = None
    sounddevice: Any = None
    clock: Callable[[], float] = time.monotonic


class CommandListener:
    def __init__(self, rules: CommandRules, settings: ListenerSettings, events: ListenerEvents,
                 engines: Engines | None = None) -> None:
        engines = engines or Engines()
        self._rules = rules
        self._settings = settings
        self._events = events
        self._vosk = engines.vosk
        self._sounddevice = engines.sounddevice
        self._clock = engines.clock
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self._vosk = self._vosk or importlib.import_module("vosk")
            self._sounddevice = self._sounddevice or importlib.import_module("sounddevice")
            listening = self._build_recognizers()
        except Exception as exc:
            raise RecognizerUnavailable(_reason(exc)) from exc
        blocks: queue.Queue[tuple[bytes, float]] = queue.Queue()
        try:
            stream = self._open_microphone(blocks)
        except Exception as exc:
            raise MicrophoneUnavailable(_reason(exc)) from exc
        with stream:
            began = self._clock()
            stall = AudioStall(now=began)
            last_heartbeat = began
            while not self._stop.is_set():
                try:
                    data, captured_at = blocks.get(timeout=self._settings.poll_seconds)
                except queue.Empty:
                    self._note_silence(stall)
                    continue
                now = self._clock()
                if stall.note_block(now=now):
                    logger.info("Voice: audio from the microphone resumed")
                    if self._events.recovered is not None:
                        self._events.recovered()
                if now - last_heartbeat >= LISTEN_HEARTBEAT_S:
                    last_heartbeat = now
                    logger.debug("Voice: listening; loudest sample since the last report: %d",
                                 listening.take_recent_level())
                heard = listening.feed(data, captured_at=captured_at)
                if heard is not None:
                    logger.log(*outcome_line(heard, now=now, rules=self._rules))
                    self._keep_a_miss(heard)
                    self._events.heard(heard)

    def _keep_a_miss(self, heard: Heard) -> None:
        missed = heard.recognition.refused_phrase or heard.recognition.unrecognized_text
        listened_to = self._events.keeps_misses is None or self._events.keeps_misses()
        if missed and heard.audio and listened_to and self._settings.miss_dir is not None:
            save_miss_audio(self._settings.miss_dir, heard.audio,
                            sample_rate=self._settings.sample_rate)

    def _note_silence(self, stall: AudioStall) -> None:
        idle = stall.note_silence(now=self._clock())
        if idle is None:
            return
        logger.warning("Voice: no audio from the microphone for %.0fs -- nothing spoken can "
                       "be heard until it comes back", idle)
        if self._events.stalled is not None:
            self._events.stalled(idle)

    def _build_recognizers(self) -> Listening:
        model = self._vosk.Model(model_name=self._settings.model_name)
        sample_rate = self._settings.sample_rate
        recognizer = self._vosk.KaldiRecognizer(
            model, sample_rate, build_grammar(self._rules.phrases))
        recognizer.SetWords(True)
        recognizer.SetMaxAlternatives(GRAMMAR_ALTERNATIVES)
        unrestricted = None
        if self._settings.caption_misses:
            unrestricted = self._vosk.KaldiRecognizer(model, sample_rate)
            unrestricted.SetWords(True)
        return Listening(self._rules, recognizer, sample_rate=sample_rate,
                         unrestricted=unrestricted, on_partial=self._events.partial)

    def _open_microphone(self, blocks: queue.Queue[tuple[bytes, float]]):
        return self._sounddevice.RawInputStream(
            samplerate=self._settings.sample_rate, blocksize=BLOCK_SAMPLES, dtype="int16", channels=1,
            device=self._device_index(),
            callback=lambda indata, _frames, _time, _status: blocks.put(
                (bytes(indata), self._clock())),
        )

    def _device_index(self) -> int | None:
        try:
            chosen = resolve_input_device(
                self._settings.device_name, sounddevice=self._sounddevice,
                probe=partial(probe_input_device, sounddevice=self._sounddevice))
        except Exception:
            logger.warning("Voice: the microphone lookup failed; using the system default",
                           exc_info=True)
            return None
        if chosen is None:
            logger.warning("Voice: no usable microphone found; using the system default")
            return None
        logger.info("Voice: listening on input device %s (%s)", chosen.index, chosen.name)
        return chosen.index
