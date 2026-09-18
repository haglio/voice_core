from __future__ import annotations

import array
import math
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import pytest

from voice_core.microphone import (
    ChosenInput,
    choose_input_device,
    loudness,
    probe_input_device,
    resolve_input_device,
)


def _device(name, *, inputs=2, hostapi=0):
    return {"name": name, "max_input_channels": inputs, "hostapi": hostapi}


def _never_probed(index):
    raise AssertionError(f"device {index} was probed")


def test_a_named_microphone_is_taken_without_listening_to_anything():
    devices = [_device("Microphone (Headset Link)"), _device("Microphone (Desk Cam 101)")]

    chosen = choose_input_device(devices, _never_probed, name="desk cam")

    assert chosen == ChosenInput(1, "Microphone (Desk Cam 101)")


def test_a_named_microphone_that_cannot_record_is_passed_over():
    devices = [_device("Desk Cam Speakers", inputs=0), _device("Microphone (Desk Cam 101)")]

    chosen = choose_input_device(devices, _never_probed, name="desk cam")

    assert chosen == ChosenInput(1, "Microphone (Desk Cam 101)")


def test_the_copy_on_the_default_host_api_is_preferred():
    devices = [
        _device("Microphone (Desk Cam 101)", hostapi=2),
        _device("Microphone (Desk Cam 101)", hostapi=0),
    ]

    chosen = choose_input_device(devices, _never_probed, name="desk cam", hostapi=0)

    assert chosen == ChosenInput(1, "Microphone (Desk Cam 101)")


def test_a_copy_on_another_host_api_beats_giving_up():
    devices = [_device("Microphone (Headset Link)", hostapi=0),
               _device("Microphone (Desk Cam 101)", hostapi=2)]

    chosen = choose_input_device(devices, _never_probed, name="desk cam", hostapi=0)

    assert chosen == ChosenInput(1, "Microphone (Desk Cam 101)")


def test_with_no_name_the_liveliest_input_beats_a_silent_default():
    devices = [_device("Dead headset mic"), _device("Real mic", inputs=1),
               _device("Speakers", inputs=0)]
    levels = {0: 0.00001, 1: 0.0003}

    chosen = choose_input_device(devices, levels.__getitem__)

    assert chosen == ChosenInput(1, "Real mic")


def test_a_device_that_will_not_open_is_skipped():
    devices = [_device("Broken"), _device("Good", inputs=1)]

    def probe(index):
        if index == 0:
            raise OSError("cannot open device")
        return 0.01

    assert choose_input_device(devices, probe) == ChosenInput(1, "Good")


def test_each_physical_microphone_is_probed_once_however_often_it_is_listed():
    devices = [_device("Desk mic"), _device("Desk mic"), _device("Desk mic")]
    probed = []

    choose_input_device(devices, lambda index: probed.append(index) or 0.01)

    assert probed == [0]


def test_only_the_default_host_api_is_listened_to():
    devices = [_device("Desk mic (kernel streaming)", hostapi=3), _device("Desk mic", hostapi=0)]
    probed = []

    chosen = choose_input_device(devices, lambda index: probed.append(index) or 0.01, hostapi=0)

    assert chosen == ChosenInput(1, "Desk mic")
    assert probed == [1]


def test_a_level_that_is_not_a_number_is_ignored():
    devices = [_device("Glitchy"), _device("Good", inputs=1)]

    chosen = choose_input_device(devices, lambda index: float("inf") if index == 0 else 0.01)

    assert chosen == ChosenInput(1, "Good")


def test_a_name_that_matches_nothing_today_falls_through_to_listening():
    devices = [_device("Desk mic")]

    assert choose_input_device(devices, lambda index: 0.01, name="headset") == ChosenInput(0, "Desk mic")


def test_a_blank_name_is_no_name():
    devices = [_device("Dead headset mic"), _device("Desk mic")]
    levels = {0: 0.0, 1: 0.01}

    assert choose_input_device(devices, levels.__getitem__, name="   ") == ChosenInput(1, "Desk mic")


def test_with_nothing_that_can_record_there_is_nothing_to_choose():
    assert choose_input_device([_device("Headphones", inputs=0)], _never_probed) is None


class _FakeSoundDevice:
    def __init__(self, devices, default_input):
        self._devices = devices
        self.default = SimpleNamespace(device=(default_input, 0))

    def query_devices(self, index=None):
        return self._devices if index is None else self._devices[index]


def test_resolving_prefers_the_named_microphone_on_the_default_devices_host_api():
    backend = _FakeSoundDevice(
        [_device("Microphone (Desk Cam 101)", hostapi=2),
         _device("Dead headset mic", hostapi=0),
         _device("Microphone (Desk Cam 101)", hostapi=0)],
        default_input=1,
    )

    chosen = resolve_input_device("desk cam", sounddevice=backend, probe=_never_probed)

    assert chosen == ChosenInput(2, "Microphone (Desk Cam 101)")


def test_with_no_default_input_no_host_api_is_ruled_out():
    backend = _FakeSoundDevice(
        [_device("Quiet mic", hostapi=3), _device("Lively mic", hostapi=0)], default_input=-1)
    levels = {0: 0.001, 1: 0.02}

    chosen = resolve_input_device(sounddevice=backend, probe=levels.__getitem__)

    assert chosen == ChosenInput(1, "Lively mic")


def test_loudness_is_the_root_mean_square_of_the_samples():
    pcm = array.array("h", [3, -4, 3, -4]).tobytes()

    assert loudness(pcm) == pytest.approx(math.sqrt(12.5))


def test_no_samples_at_all_are_silence():
    assert loudness(b"") == 0.0


@contextmanager
def _a_stream_that_delivers_one_second(opened, **kwargs):
    opened.append(kwargs)
    one_second = array.array("h", [100, -100] * (kwargs["samplerate"] // 2)).tobytes()
    kwargs["callback"](one_second, len(one_second) // 2, None, None)
    yield


def test_a_probe_listens_at_the_devices_own_rate_and_reports_how_loud_it_was():
    opened = []
    backend = _FakeSoundDevice([{**_device("Desk mic"), "default_samplerate": 48000.0}], 0)
    backend.RawInputStream = partial(_a_stream_that_delivers_one_second, opened)

    level = probe_input_device(0, sounddevice=backend)

    assert level == pytest.approx(100.0)
    assert (opened[0]["device"], opened[0]["samplerate"], opened[0]["dtype"]) == (0, 48000, "int16")
