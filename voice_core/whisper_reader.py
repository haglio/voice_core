from __future__ import annotations

import importlib
import logging
import math
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Prompted with an utterance's own candidates, "base" chose as well as "small" on the owner's
# recordings (7 of 10 missed commands either way, 3 against 9 invented in 930 utterances of
# ordinary talk) in a quarter of the time: about a third of a second a reading on this machine.
COMMAND_MODEL = "base"

# Taking his sentences down is where "small" earns its second: measured against the engine his
# dictation apps use, its worst quarter of 24 utterances came within 0.86 of it, base within 0.59.
DICTATION_MODEL = "small"

# Prompted in German, "small" read the owner's own "genau" as itself in all three kept clips of him
# saying it, where "base" wrote "Guten Appetit" (2026-10-07), so a phrase said in another language
# is read by it.
OTHER_LANGUAGE_MODEL = "small"

_FULL_LEVEL = 0.95
_SILENCE = 1e-4

# The longest phrase an app had was 10 tokens, and a reading that runs on past a few tellings of
# one is whisper repeating itself, so it is stopped there.
_COMMAND_TOKENS = 48

# Whisper pads whatever it reads to 30 s, so reading a one-second command spent nine tenths of its
# time on silence: a 3 s window read his Genau clips in 0.15 s against 1.0 s, and read all four.
_FRAMES_A_SECOND = 100
_SHORTEST_WINDOW = 3 * _FRAMES_A_SECOND
_LONGEST_WINDOW = 30 * _FRAMES_A_SECOND
_QUIET_AFTER = _FRAMES_A_SECOND // 2

# Whisper's own bars for calling a reading silence.
_NO_SPEECH_BAR = 0.6
_LOG_PROB_BAR = -1.0


@dataclass(frozen=True)
class Reading:
    text: str
    no_speech_prob: float
    avg_logprob: float

    def heard_as_silence(self) -> bool:
        return self.no_speech_prob > _NO_SPEECH_BAR and self.avg_logprob < _LOG_PROB_BAR


def _window_frames(samples: int) -> int:
    frames = math.ceil(samples * _FRAMES_A_SECOND / 16000) + _QUIET_AFTER
    return min(_LONGEST_WINDOW, max(_SHORTEST_WINDOW, frames))


class _FasterWhisper:
    def __init__(self, model) -> None:
        self._model = model
        self._tokenizers: dict[str, Any] = {}

    def transcribe(self, audio, **options):
        return self._model.transcribe(audio, **options)

    def read_window(self, audio, *, language: str, hint: str, frames: int,
                    max_new_tokens: int) -> Reading:
        engine = faster_whisper("faster_whisper.transcribe")
        model, tokenizer = self._model, self._tokenizer(language)
        features = faster_whisper("faster_whisper.audio").pad_or_trim(
            model.feature_extractor(audio), frames)
        prompt = model.get_prompt(tokenizer, tokenizer.encode(" " + hint.strip()) if hint else [],
                                  without_timestamps=True)
        [result] = model.model.generate(
            model.encode(features), [prompt], beam_size=5, max_length=len(prompt) + max_new_tokens,
            return_scores=True, return_no_speech_prob=True, suppress_blank=True,
            suppress_tokens=engine.get_suppressed_tokens(tokenizer, [-1]))
        tokens = result.sequences_ids[0]
        return Reading(tokenizer.decode(tokens).strip(), result.no_speech_prob,
                       result.scores[0] * len(tokens) / (len(tokens) + 1))

    def _tokenizer(self, language: str):
        if language not in self._tokenizers:
            self._tokenizers[language] = faster_whisper("faster_whisper.tokenizer").Tokenizer(
                self._model.hf_tokenizer, self._model.model.is_multilingual, task="transcribe",
                language=language)
        return self._tokenizers[language]


def faster_whisper(module: str = "faster_whisper"):
    # faster-whisper needs no torch but imports any it finds, and a torch that loads
    # fine alone can die initializing c10.dll once Qt's DLLs are in the process
    # (WinError 1114), taking every reading with it. A torch already imported stays.
    sys.modules.setdefault("torch", None)
    return importlib.import_module(module)


def load_faster_whisper(model_size: str):
    whisper_model = faster_whisper().WhisperModel
    # On the CPU by design: the one GPU is for pictures and Funestras. A reading took a median
    # 0.43 s on two threads, 0.33 s on four and 0.27 s on eight (this machine, 2026-09-18):
    # four is most of the gain and leaves the Funestras their cores.
    on_this_machine = partial(whisper_model, model_size, device="cpu", compute_type="int8",
                              cpu_threads=4)
    try:
        return _FasterWhisper(on_this_machine(local_files_only=True))
    except FileNotFoundError:
        return _FasterWhisper(on_this_machine(local_files_only=False))


class WhisperReader:
    """Reads one utterance: ``reader(pcm, hint)``.

    As recorded, for choosing among a command's candidates -- lifting the level or filtering
    for speech cost that reading a clip or two in ten. *for_dictation*, a quiet microphone is
    lifted to full level and whisper skips what is not speech, which is what steadied sentences.
    """

    def __init__(self, *, load: Callable[[], Any] | None = None, for_dictation: bool = False,
                 language: str = "en", clock: Callable[[], float] = time.monotonic) -> None:
        size = (DICTATION_MODEL if for_dictation
                else COMMAND_MODEL if language == "en" else OTHER_LANGUAGE_MODEL)
        self._load = load or partial(load_faster_whisper, size)
        self._for_dictation = for_dictation
        self._language = language
        self._clock = clock
        self._model = None
        self._loading = threading.Lock()

    def load_ahead(self) -> None:
        threading.Thread(target=self._load_or_say_why_not, name="whisper-load-ahead",
                         daemon=True).start()

    def preload(self) -> None:
        with self._loading:
            self._load_once()

    def _load_or_say_why_not(self) -> None:
        with self._loading:
            try:
                self._load_once()
            except Exception:
                logger.warning("Voice: whisper did not load ahead; the first reading tries again",
                               exc_info=True)

    def _load_once(self) -> None:
        if self._model is None:
            began = self._clock()
            self._model = self._load()
            logger.info("Voice: whisper loaded in %.1fs", self._clock() - began)

    def __call__(self, pcm: bytes, hint: str) -> str:
        if self._for_dictation:
            return self._take_down(pcm, hint)
        reading = self._read_command(pcm, hint, lifted=False)
        return "" if reading.heard_as_silence() else reading.text

    def read_closely(self, pcm: bytes, hint: str) -> str:
        return self._read_command(pcm, hint, lifted=True).text

    def _take_down(self, pcm: bytes, hint: str) -> str:
        self.preload()
        segments, _ = self._model.transcribe(
            _samples(pcm, lifted=True), language=self._language, initial_prompt=hint or None,
            condition_on_previous_text=False, vad_filter=True)
        return " ".join(part for part in (segment.text.strip() for segment in segments) if part)

    def _read_command(self, pcm: bytes, hint: str, *, lifted: bool) -> Reading:
        self.preload()
        audio = _samples(pcm, lifted=lifted)
        return self._model.read_window(audio, language=self._language, hint=hint,
                                       frames=_window_frames(len(audio)),
                                       max_new_tokens=_COMMAND_TOKENS)


def _samples(pcm: bytes, *, lifted: bool) -> np.ndarray:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if lifted:
        audio = audio / max(float(np.max(np.abs(audio), initial=0.0)), _SILENCE) * _FULL_LEVEL
    return audio
