from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from voice_core.capture import AudioStall
from voice_core.commands import CommandRules
from voice_core.listening import Heard, Listening, PausedListening, Recognizers, outcome_line
from voice_core.microphone import probe_input_device, resolve_input_device
from voice_core.miss_clips import save_miss_audio
from voice_core.pauses import PauseSegmenter
from voice_core.readings import build_grammar
from voice_core.second_opinion import settle

logger = logging.getLogger(__name__)

BLOCK_SAMPLES = 8000

# Vosk's grammar restricts the vocabulary, not the phrases: it decodes any
# sequence of the phrases' words, so the right phrase often sits in the rankings
# under a sequence that is no command.
GRAMMAR_ALTERNATIVES = 5

# Without it, a session where every phrase missed and one where the microphone
# was dead leave the same log.
LISTEN_HEARTBEAT_S = 60.0

# How long a closing listener waits for a reading already under way.
SECOND_OPINION_PATIENCE_S = 5.0


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
class PauseSettings:
    """Where an utterance ends for an app that takes dictation, in 16-bit sample units.

    The owner's microphone is quiet -- his speech runs 60 to 600 a frame, against digital
    silence -- so the floor is the one his dictation app settled on over replayed sessions
    (0.0008 of full scale), not the 0.008 Origenerator had, which sat above most of his words.
    Half a second of quiet ends an utterance; under 180 ms of sound holds no word."""

    floor: float = 26.0
    ratio: float = 2.5
    calibration_frames: int = 15
    hangover_frames: int = 17
    min_speech_frames: int = 6
    frame_samples: int = 480  # 30 ms at 16 kHz


@dataclass(frozen=True)
class ListenerSettings:
    model_name: str
    device_name: str | None = None
    sample_rate: int = 16000
    caption_misses: bool = False
    # How much of an utterance's audio is kept for a second engine and for a miss's clip:
    # four seconds holds any command, a dictated sentence wants more.
    kept_seconds: float = 4.0
    # Set by an app that takes dictation: utterances end where the speaker pauses, since vosk
    # ends one only when its grammar has nothing more to say.
    pauses: PauseSettings | None = None
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
    # Whether the room is being listened to: what a muted app mishears is not kept.
    keeps_misses: Callable[[], bool] | None = None
    # For an app that takes dictation: the words of an utterance that was no command, handed
    # over even when there were none to read, so the app can say it caught nothing.
    speech: Callable[[str, Heard], None] | None = None


@dataclass(frozen=True)
class Engines:
    vosk: Any = None
    sounddevice: Any = None
    clock: Callable[[], float] = time.monotonic
    # Reads an utterance's audio given a hint of phrases; see second_opinion.settle.
    second_opinion: Callable[[bytes, str], str] | None = None
    # Reads an utterance's audio as ordinary speech, given the app's standing hint.
    take_down: Callable[[bytes, str], str] | None = None


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
        self._second_opinion = engines.second_opinion
        self._take_down = engines.take_down
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
        with stream, self._delivery() as deliver:
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
                    deliver(heard)

    @contextmanager
    def _delivery(self):
        """How a settled utterance reaches the app: at once, or -- a second engine taking
        its second over each -- from one thread of its own, in the order they were spoken."""
        if self._second_opinion is None and self._take_down is None:
            yield self._deliver
            return
        waiting: queue.Queue[Heard | None] = queue.Queue()

        def in_order() -> None:
            self._load_the_engines_ahead()
            while (heard := waiting.get()) is not None:
                self._deliver(self._settled(heard))

        worker = threading.Thread(target=in_order, name="voice-second-opinion", daemon=True)
        worker.start()
        try:
            yield waiting.put
        finally:
            waiting.put(None)
            worker.join(timeout=SECOND_OPINION_PATIENCE_S)

    def _load_the_engines_ahead(self) -> None:
        for engine in (self._second_opinion, self._take_down):
            preload = getattr(engine, "preload", None)
            if preload is None:
                continue
            try:
                preload()
            except Exception:
                logger.exception("Voice: the second engine did not load")

    def _settled(self, heard: Heard) -> Heard:
        if self._second_opinion is None:
            return heard
        try:
            return settle(heard, rules=self._rules, read=self._second_opinion)
        except Exception:
            logger.exception("Voice: the second engine failed; the first engine's word stands")
            return heard

    def _deliver(self, heard: Heard) -> None:
        logger.log(*outcome_line(heard, now=self._clock(), rules=self._rules))
        self._keep_a_miss(heard)
        self._events.heard(heard)
        self._hand_over_speech(heard)

    def _hand_over_speech(self, heard: Heard) -> None:
        wanted = self._events.speech is not None and self._take_down is not None
        if not wanted or heard.recognition.phrase or self._read_out_of_silence(heard):
            return
        try:
            words = self._take_down(heard.audio, self._settings.speech_hint)
        except Exception:
            logger.exception("Voice: the utterance could not be taken down")
            return
        self._events.speech(words, heard)

    def _read_out_of_silence(self, heard: Heard) -> bool:
        # Vosk ends utterances in a silent room too, and whisper makes sentences of those.
        # Where pauses end them, the pause detector has already said somebody spoke -- and
        # on the owner's quiet microphone one real stretch in seven peaks under this bar.
        return self._settings.pauses is None and heard.peak < self._rules.silent_peak

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
        pauses = self._settings.pauses
        if pauses is not None:
            return PausedListening(
                self._rules, Recognizers(recognizer),
                PauseSegmenter(floor=pauses.floor, ratio=pauses.ratio,
                               calibration_frames=pauses.calibration_frames,
                               hangover_frames=pauses.hangover_frames,
                               min_speech_frames=pauses.min_speech_frames),
                sample_rate=sample_rate)
        return Listening(self._rules, Recognizers(recognizer, unrestricted),
                         sample_rate=sample_rate, on_partial=self._events.partial,
                         kept_blocks=round(self._settings.kept_seconds * sample_rate / BLOCK_SAMPLES))

    def _open_microphone(self, blocks: queue.Queue[tuple[bytes, float]]):
        return self._sounddevice.RawInputStream(
            samplerate=self._settings.sample_rate, blocksize=self._block_samples(), dtype="int16",
            channels=1,
            device=self._device_index(),
            callback=lambda indata, _frames, _time, _status: blocks.put(
                (bytes(indata), self._clock())),
        )

    def _block_samples(self) -> int:
        pauses = self._settings.pauses
        return BLOCK_SAMPLES if pauses is None else pauses.frame_samples

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
