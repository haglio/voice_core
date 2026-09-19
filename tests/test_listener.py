from __future__ import annotations

import array
import json
import logging
import sys
import wave
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from voice_core.commands import CommandRules
from voice_core.listener import (
    CommandListener,
    Engines,
    ListenerEvents,
    ListenerSettings,
    MicrophoneUnavailable,
    PauseSettings,
    RecognizerUnavailable,
    why_unavailable,
)

RULES = CommandRules(phrases=frozenset({"next", "left next"}), never_rescued=lambda phrase: False)
STALLED_FOR = 11.0
HALF_SECOND = array.array("h", [2000, -2000] * 4000).tobytes()


class _Recognizer:
    def __init__(self, model, sample_rate, grammar=None, *, reading="next", settles=True):
        self.grammar = grammar
        self._reading = reading
        self._settles = settles
        self.words = False
        self.alternatives = 0
        self._fed = 0

    def SetWords(self, enable):  # noqa: N802 -- vosk's spelling
        self.words = enable

    def SetMaxAlternatives(self, count):  # noqa: N802
        self.alternatives = count

    def AcceptWaveform(self, data):  # noqa: N802
        self._fed += 1
        return self._settles and self.grammar is not None and self._fed == 2  # noqa: PLR2004

    def Result(self):  # noqa: N802
        return json.dumps({"alternatives": [{"text": self._reading, "confidence": 1.0}]})

    def FinalResult(self):  # noqa: N802
        return self.Result() if self.grammar is not None else ""

    def PartialResult(self):  # noqa: N802
        return json.dumps({"partial": "ne"})


def _vosk(built, reading="next", settles=True):
    def recognizer(*args):
        built.append(_Recognizer(*args, reading=reading, settles=settles))
        return built[-1]
    return SimpleNamespace(Model=lambda model_name: model_name, KaldiRecognizer=recognizer)


def _sounddevice(opened, blocks):
    devices = [{"name": "Desk mic", "max_input_channels": 1, "hostapi": 0}]

    @contextmanager
    def stream(**kwargs):
        opened.append(kwargs)
        for block in blocks:
            kwargs["callback"](block, len(block) // 2, None, None)
        yield

    return SimpleNamespace(
        default=SimpleNamespace(device=(0, 0)),
        query_devices=lambda index=None: devices if index is None else devices[index],
        RawInputStream=stream,
    )


SETTINGS = ListenerSettings(model_name="a-model", device_name="desk", poll_seconds=0.001)
TWO_BLOCKS = [HALF_SECOND, HALF_SECOND]


def _listener(*, settings=SETTINGS, engines, keeps_misses=None, **events):
    """A listener that stops itself at whichever event fires first."""
    def stopping(wanted=None):
        def event(*args):
            if wanted is not None:
                wanted(*args)
            listener.stop()
        return event

    listener = CommandListener(
        RULES, settings,
        ListenerEvents(**{"heard": stopping(), "keeps_misses": keeps_misses,
                          **{name: stopping(wanted) for name, wanted in events.items()}}),
        engines)
    return listener


def test_running_hears_what_the_named_microphone_delivers_until_stopped():
    built, opened, heard = [], [], []

    _listener(heard=heard.append,
              engines=Engines(_vosk(built), _sounddevice(opened, TWO_BLOCKS))).run()

    assert [one.recognition.phrase for one in heard] == ["next"]
    assert (opened[0]["device"], opened[0]["samplerate"], opened[0]["blocksize"],
            opened[0]["dtype"], opened[0]["channels"]) == (0, 16000, 8000, "int16", 1)


def _run_until_heard(settings=SETTINGS, **events):
    built = []
    _listener(settings=settings, engines=Engines(_vosk(built), _sounddevice([], TWO_BLOCKS)),
              **events).run()
    return built


def test_the_grammar_recognizer_is_asked_for_its_ranked_readings_over_the_apps_phrases():
    grammar_recognizer = _run_until_heard()[0]

    assert json.loads(grammar_recognizer.grammar) == ["left next", "next", "[unk]"]
    assert (grammar_recognizer.words, grammar_recognizer.alternatives) == (True, 5)


def test_misses_are_captioned_by_an_unrestricted_recognizer_only_when_asked_for():
    without = _run_until_heard()
    with_captions = _run_until_heard(replace(SETTINGS, caption_misses=True))

    assert [recognizer.grammar is None for recognizer in without] == [False]
    assert [(recognizer.grammar is None, recognizer.words) for recognizer in with_captions] == [
        (False, True), (True, True)]


REFUSES = Mock(side_effect=OSError("no such thing"))


def test_a_model_that_will_not_load_and_a_microphone_that_will_not_open_fail_by_name():
    no_model = SimpleNamespace(Model=REFUSES, KaldiRecognizer=_Recognizer)
    no_microphone = _sounddevice([], [])
    no_microphone.RawInputStream = REFUSES

    with pytest.raises(RecognizerUnavailable, match="no such thing"):
        _listener(engines=Engines(no_model, _sounddevice([], []))).run()
    with pytest.raises(MicrophoneUnavailable, match="no such thing"):
        _listener(engines=Engines(_vosk([]), no_microphone)).run()


def test_a_microphone_lookup_that_fails_falls_back_to_the_systems_default():
    opened = []
    backend = _sounddevice(opened, TWO_BLOCKS)
    backend.query_devices = REFUSES

    _listener(engines=Engines(_vosk([]), backend)).run()

    assert opened[0]["device"] is None


def test_a_microphone_that_goes_quiet_is_reported_once_with_how_long_it_has_been():
    stalls = []
    ticks = iter([0.0, 4.0, 11.0, 12.0])

    _listener(stalled=stalls.append,
              engines=Engines(_vosk([]), _sounddevice([], []), clock=lambda: next(ticks))).run()

    assert stalls == [STALLED_FOR]


def test_audio_coming_back_after_a_stall_is_news_of_its_own():
    opened, recovered = [], []
    ticks = iter([0.0, 11.0, "the microphone comes back"])

    def clock():
        tick = next(ticks, 12.0)
        if tick == "the microphone comes back":
            opened[0]["callback"](HALF_SECOND, 4000, None, None)
            return 12.0
        return tick

    listener = CommandListener(
        RULES, SETTINGS,
        ListenerEvents(heard=print, recovered=lambda: (recovered.append(True), listener.stop())),
        Engines(_vosk([]), _sounddevice(opened, []), clock=clock))
    listener.run()

    assert recovered == [True]


def test_the_words_still_forming_reach_whoever_asked_for_them():
    forming = []

    _run_until_heard(partial=forming.append)

    assert forming == ["ne"]


def test_once_a_minute_the_log_says_how_loud_the_microphone_has_been(caplog):
    ticks = iter([0.0, 0.0, 0.0, 1.0, 61.0])  # two blocks captured, the start, then each block read

    with caplog.at_level(logging.DEBUG, logger="voice_core.listener"):
        _listener(engines=Engines(_vosk([]), _sounddevice([], TWO_BLOCKS),
                                  clock=lambda: next(ticks, 61.0))).run()

    assert "Voice: listening; loudest sample since the last report: 2000" in caplog.messages


def test_with_no_engines_handed_in_the_installed_ones_are_used(monkeypatch):
    built, opened = [], []
    monkeypatch.setitem(sys.modules, "vosk", _vosk(built))
    monkeypatch.setitem(sys.modules, "sounddevice", _sounddevice(opened, TWO_BLOCKS))

    _listener(engines=None).run()

    assert (len(built), len(opened)) == (1, 1)


def test_why_voice_cannot_run_is_the_import_that_failed_or_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "vosk", _vosk([]))
    monkeypatch.setitem(sys.modules, "sounddevice", _sounddevice([], []))
    assert why_unavailable() == ""

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert "sounddevice" in why_unavailable()


def _a_miss(tmp_path, **events):
    built = []
    engines = Engines(_vosk(built, reading="left net"), _sounddevice([], TWO_BLOCKS))
    _listener(settings=replace(SETTINGS, miss_dir=tmp_path / "misses"), engines=engines,
              **events).run()
    return sorted((tmp_path / "misses").glob("*.wav"))


def test_the_audio_of_a_miss_is_kept_where_the_app_said(tmp_path):
    [clip] = _a_miss(tmp_path)

    with wave.open(str(clip), "rb") as kept:
        assert kept.getnframes() == len(HALF_SECOND + HALF_SECOND) // 2


def test_nothing_is_kept_of_a_miss_while_the_app_is_not_listening(tmp_path):
    assert _a_miss(tmp_path, keeps_misses=lambda: False) == []


def test_a_miss_is_kept_while_the_app_says_it_is_listening(tmp_path):
    assert len(_a_miss(tmp_path, keeps_misses=lambda: True)) == 1


def test_an_app_that_names_nowhere_for_misses_has_none_of_its_audio_written(monkeypatch):
    # Origenerator promises that what is said to it never reaches the disk.
    kept = Mock()
    monkeypatch.setattr("voice_core.listener.save_miss_audio", kept)
    heard = []

    _listener(heard=heard.append,
              engines=Engines(_vosk([], reading="left net"), _sounddevice([], TWO_BLOCKS))).run()

    assert [one.recognition.unrecognized_text for one in heard] == ["left net"]
    kept.assert_not_called()


def test_what_the_microphone_delivered_is_kept_though_the_driver_reuses_its_buffer():
    heard = []
    quiet = bytes(len(HALF_SECOND))

    @contextmanager
    def stream(**kwargs):
        buffer = bytearray(HALF_SECOND)
        kwargs["callback"](buffer, len(buffer) // 2, None, None)
        buffer[:] = quiet
        kwargs["callback"](buffer, len(buffer) // 2, None, None)
        yield

    backend = _sounddevice([], [])
    backend.RawInputStream = stream

    _listener(heard=heard.append, engines=Engines(_vosk([]), backend)).run()

    assert [one.audio for one in heard] == [HALF_SECOND + quiet]


def test_a_second_engine_settles_each_utterance_before_the_app_hears_of_it(tmp_path):
    heard, asked = [], []

    def second_opinion(audio, hint):
        asked.append((len(audio), hint))
        return "and then"

    _listener(settings=replace(SETTINGS, miss_dir=tmp_path), heard=heard.append,
              engines=Engines(_vosk([]), _sounddevice([], TWO_BLOCKS),
                              second_opinion=second_opinion)).run()

    assert asked == [(len(HALF_SECOND + HALF_SECOND), "next")]
    assert [one.recognition.unconfirmed_phrase for one in heard] == ["next"]
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_a_second_engine_that_fails_leaves_the_first_engines_word_standing(caplog):
    heard = []
    failing = Mock(side_effect=ModuleNotFoundError("No module named 'faster_whisper'"))

    with caplog.at_level(logging.ERROR, logger="voice_core.listener"):
        _listener(heard=heard.append,
                  engines=Engines(_vosk([]), _sounddevice([], TWO_BLOCKS),
                                  second_opinion=failing)).run()

    assert [one.recognition.phrase for one in heard] == ["next"]
    assert "No module named 'faster_whisper'" in caplog.text


def test_a_second_engine_that_can_load_ahead_is_loaded_before_the_first_utterance_needs_it():
    order = []

    class _Reader:
        def preload(self):
            order.append("loaded")

        def __call__(self, audio, hint):
            order.append("asked")
            return hint

    _listener(engines=Engines(_vosk([]), _sounddevice([], TWO_BLOCKS),
                              second_opinion=_Reader())).run()

    assert order == ["loaded", "asked"]


def test_an_utterance_that_is_no_command_is_taken_down_for_an_app_that_wants_speech():
    spoken, asked = [], []

    def take_down(audio, hint):
        asked.append((len(audio), hint))
        return "make the sky darker"

    _listener(settings=replace(SETTINGS, speech_hint="Voice requests: request, over."),
              speech=lambda text, heard: spoken.append((text, heard.recognition.unrecognized_text)),
              engines=Engines(_vosk([], reading="left net"), _sounddevice([], TWO_BLOCKS),
                              take_down=take_down)).run()

    assert asked == [(len(HALF_SECOND + HALF_SECOND), "Voice requests: request, over.")]
    assert spoken == [("make the sky darker", "left net")]


def _spoken(reading, take_down, blocks=None):
    spoken = []
    _listener(speech=lambda text, heard: spoken.append(text),
              engines=Engines(_vosk([], reading=reading),
                              _sounddevice([], blocks or TWO_BLOCKS), take_down=take_down)).run()
    return spoken


def test_a_command_is_not_also_taken_down_as_speech():
    take_down = Mock(return_value="next")

    assert _spoken("next", take_down) == []
    take_down.assert_not_called()


def test_what_was_read_out_of_silence_is_not_taken_down():
    take_down = Mock(return_value="thank you")

    assert _spoken("left net", take_down, blocks=[bytes(len(HALF_SECOND))] * 2) == []
    take_down.assert_not_called()


def test_a_take_down_with_no_words_in_it_is_dropped():
    assert _spoken("left net", lambda audio, hint: " . . . ") == []


def test_an_engine_that_cannot_take_an_utterance_down_is_logged_and_nothing_is_said(caplog):
    with caplog.at_level(logging.ERROR, logger="voice_core.listener"):
        spoken = _spoken("left net", Mock(side_effect=OSError("model gone")))

    assert spoken == []
    assert "could not be taken down" in caplog.text


def test_an_utterance_keeps_as_many_seconds_of_audio_as_the_app_asks_for():
    heard = []

    _listener(settings=replace(SETTINGS, kept_seconds=0.5), heard=heard.append,
              engines=Engines(_vosk([]), _sounddevice([], TWO_BLOCKS))).run()

    assert [len(one.audio) for one in heard] == [len(HALF_SECOND)]


def test_an_app_that_takes_dictation_hears_utterances_end_where_the_speaker_pauses():
    frame = array.array("h", [2000, -2000] * 240).tobytes()
    quiet = bytes(len(frame))
    opened, heard = [], []
    pauses = PauseSettings(floor=262, ratio=2.0, calibration_frames=1, hangover_frames=2,
                           min_speech_frames=2)

    _listener(settings=replace(SETTINGS, pauses=pauses), heard=heard.append,
              engines=Engines(_vosk([], settles=False),
                              _sounddevice(opened, [quiet, frame, frame, quiet, quiet]))).run()

    assert opened[0]["blocksize"] == len(frame) // 2
    assert [(one.recognition.phrase, one.audio) for one in heard] == [
        ("next", frame + frame + quiet + quiet)]
