from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from app_support import haglio_settings

from voice_core.capture import AudioStall
from voice_core.commands import CommandRules
from voice_core.listening import Heard, Listening, Recognizers, outcome_line
from voice_core.microphone import probe_input_device, resolve_input_device
from voice_core.miss_clips import save_miss_audio
from voice_core.old_names import answers_to_old_names
from voice_core.pauses import PauseSegmenter, PauseSettings
from voice_core.readings import build_grammar
from voice_core.sound_library import load_sounddevice
from voice_core.speech import hears_speech
from voice_core.verifying_listener import settle
from voice_core.whisper_reader import WhisperReader

logger = logging.getLogger(__name__)

# Vosk's grammar restricts the vocabulary, not the phrases: it decodes any
# sequence of the phrases' words, so the right phrase often sits in the rankings
# under a sequence that is no command.
GRAMMAR_ALTERNATIVES = 5

# Without it, a session where every phrase missed and one where the microphone
# was dead leave the same log.
LISTEN_HEARTBEAT_S = 60.0

# How long a closing listener waits for a reading already under way.
READING_PATIENCE_S = 5.0


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
        load_sounddevice()
        importlib.import_module("vosk")
    except Exception as exc:
        return _reason(exc)
    return ""


@dataclass(frozen=True)
class ListenerSettings:
    model_name: str = field(default_factory=lambda: haglio_settings.value("vosk_model"))
    device_name: str | None = field(default_factory=lambda: haglio_settings.value("microphone"))
    sample_rate: int = 16000
    caption_misses: bool = False
    pauses: PauseSettings = field(default_factory=PauseSettings)
    miss_dir: Path | None = None
    # What the engine that takes speech down is told to expect, beyond ordinary words.
    speech_hint: str = ""
    poll_seconds: float = 0.5


@dataclass(frozen=True)
class ListenerEvents:
    heard: Callable[[Heard], None]
    partial: Callable[[str], None] | None = None
    stalled: Callable[[float], None] | None = None
    recovered: Callable[[], None] | None = None
    listening: Callable[[], None] | None = None
    # Whether the room is being listened to: what a muted app mishears is not kept.
    keeps_misses: Callable[[], bool] | None = None
    speech: Callable[[str, Heard], None] | None = None
    wants_words: Callable[[Heard], bool] | None = None


@answers_to_old_names(second_opinion="verifying_listener",
                      second_opinion_in="verifying_listener_in")
@dataclass(frozen=True)
class Engines:
    vosk: Any = None
    sounddevice: Any = None
    clock: Callable[[], float] = time.monotonic
    verifying_listener: Callable[[bytes, str], str] | None = None
    verifying_listener_in: Mapping[str, Callable[[bytes, str], str]] = field(default_factory=dict)
    # Reads an utterance's audio as ordinary speech, given the app's standing hint.
    take_down: Callable[[bytes, str], str] | None = None
    # Says of each frame whether somebody is speaking, so an utterance ends when the speaker
    # pauses even in a room loud enough to hold it open (speech.hears_speech).
    hears_speech: Callable[[bytes], bool] | None = None


@answers_to_old_names(second_opinion="verifying_listener")
def engines_for(rules: CommandRules, events: ListenerEvents, *,
                clock: Callable[[], float] = time.monotonic,
                verifying_listener: Callable[[bytes, str], str] | None = None) -> Engines:
    return Engines(clock=clock,
                   verifying_listener=verifying_listener or WhisperReader(),
                   verifying_listener_in={language: WhisperReader(language=language)
                                          for language in rules.other_languages()},
                   take_down=WhisperReader(for_dictation=True) if events.speech else None,
                   hears_speech=hears_speech())


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
        self._verifying_listener = engines.verifying_listener
        self._verifying_listener_in = dict(engines.verifying_listener_in)
        self._take_down = engines.take_down
        self._hears_speech = engines.hears_speech
        self._readers = tuple(
            reader for reader in (engines.verifying_listener, engines.take_down,
                                  *engines.verifying_listener_in.values())
            if reader is not None)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self._start_loading_the_readers()
        try:
            self._vosk = self._vosk or importlib.import_module("vosk")
            self._sounddevice = self._sounddevice or load_sounddevice()
            listening = self._build_recognizers()
        except Exception as exc:
            raise RecognizerUnavailable(_reason(exc)) from exc
        blocks: queue.Queue[tuple[bytes, float]] = queue.Queue()
        try:
            stream = self._open_microphone(blocks)
        except Exception as exc:
            raise MicrophoneUnavailable(_reason(exc)) from exc
        with stream, self._delivery() as deliver:
            if self._events.listening is not None:
                self._events.listening()
            began = self._clock()
            stall = AudioStall(now=began)
            last_heartbeat = began
            while not self._stop.is_set():
                try:
                    data, started_at = blocks.get(timeout=self._settings.poll_seconds)
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
                heard = listening.feed(data, started_at=started_at)
                if heard is not None:
                    deliver(heard)

    @contextmanager
    def _delivery(self):
        """How a settled utterance reaches the app: at once, or -- readers taking their
        time over each -- from one thread of its own, in the order they were spoken."""
        if not self._readers:
            yield self._deliver
            return
        waiting: queue.Queue[Heard | None] = queue.Queue()

        def in_order() -> None:
            self._finish_loading_the_readers()
            while (heard := waiting.get()) is not None:
                self._deliver(self._settled(heard))

        worker = threading.Thread(target=in_order, name="voice-readers", daemon=True)
        worker.start()
        try:
            yield waiting.put
        finally:
            waiting.put(None)
            worker.join(timeout=READING_PATIENCE_S)

    def _start_loading_the_readers(self) -> None:
        for reader in self._readers:
            load_ahead = getattr(reader, "load_ahead", None)
            if load_ahead is not None:
                load_ahead()

    def _finish_loading_the_readers(self) -> None:
        for reader in self._readers:
            preload = getattr(reader, "preload", None)
            if preload is None:
                continue
            try:
                preload()
            except Exception:
                logger.exception("Voice: a reader did not load")

    def _settled(self, heard: Heard) -> Heard:
        if self._verifying_listener is None:
            return heard
        try:
            return settle(heard, rules=self._rules, read=self._verifying_listener,
                          read_closely=getattr(self._verifying_listener, "read_closely", None),
                          readers_in=self._verifying_listener_in)
        except Exception:
            logger.exception(
                "Voice: the verifying listener failed; the first listener's word stands")
            return heard

    def _deliver(self, heard: Heard) -> None:
        logger.log(*outcome_line(heard, now=self._clock(), rules=self._rules))
        self._keep_a_miss(heard)
        self._events.heard(heard)
        self._hand_over_speech(heard)

    def _takes_dictation(self) -> bool:
        return self._events.speech is not None and self._take_down is not None

    def _wants_the_words_of(self, heard: Heard) -> bool:
        if self._events.wants_words is None:
            return not heard.recognition.phrase
        return self._events.wants_words(heard)

    def _hand_over_speech(self, heard: Heard) -> None:
        if not self._takes_dictation() or not self._wants_the_words_of(heard):
            return
        try:
            words = self._take_down(heard.audio, self._settings.speech_hint)
        except Exception:
            logger.exception("Voice: the utterance could not be taken down")
            return
        self._events.speech(words, heard)

    def _keep_a_miss(self, heard: Heard) -> None:
        recognition = heard.recognition
        missed = (recognition.refused_phrase or recognition.unrecognized_text
                  or recognition.unconfirmed_phrase)
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
        return Listening(self._rules, Recognizers(recognizer, unrestricted, sample_rate),
                         PauseSegmenter(self._settings.pauses, hears_speech=self._hears_speech),
                         on_partial=self._events.partial,
                         for_dictation=self._takes_dictation()
                         and self._events.wants_words is None)

    def _open_microphone(self, blocks: queue.Queue[tuple[bytes, float]]):
        return self._sounddevice.RawInputStream(
            samplerate=self._settings.sample_rate, blocksize=self._settings.pauses.frame_samples,
            dtype="int16", channels=1, device=self._device_index(),
            callback=lambda indata, frames, _time, _status: blocks.put(
                (bytes(indata), self._clock() - frames / self._settings.sample_rate)),
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
