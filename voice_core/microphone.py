from __future__ import annotations

import array
import logging
import math
import queue
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChosenInput:
    index: int
    name: str


def choose_input_device(devices, probe, *, name=None, hostapi=None):
    inputs = [(index, device) for index, device in enumerate(devices)
              if device.get("max_input_channels", 0) > 0]
    return _named(inputs, name, hostapi) or _liveliest(
        [(index, device) for index, device in inputs
         if hostapi is None or device.get("hostapi") == hostapi],
        probe)


def _named(inputs, name, hostapi):
    wanted = (name or "").strip().lower()
    if not wanted:
        return None
    matches = [(index, device) for index, device in inputs if wanted in device["name"].lower()]
    on_default_api = [match for match in matches if match[1].get("hostapi") == hostapi]
    for index, device in on_default_api or matches:
        return ChosenInput(index, device["name"])
    return None


def _liveliest(inputs, probe):
    best = None
    best_level = None
    probed_names = set()
    for index, device in inputs:
        if device["name"] in probed_names:
            continue
        probed_names.add(device["name"])
        try:
            level = probe(index)
        except Exception:
            logger.debug("Input device %d could not be probed.", index, exc_info=True)
            continue
        if not math.isfinite(level):
            continue
        if best_level is None or level > best_level:
            best, best_level = ChosenInput(index, device["name"]), level
    return best


def resolve_input_device(name=None, *, sounddevice, probe):
    default_input = sounddevice.default.device[0]
    has_default = isinstance(default_input, int) and default_input >= 0
    hostapi = sounddevice.query_devices(default_input)["hostapi"] if has_default else None
    return choose_input_device(sounddevice.query_devices(), probe, name=name, hostapi=hostapi)


def loudness(pcm):
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


PROBE_SECONDS = 0.4


def probe_input_device(index, *, sounddevice):
    # At the device's own rate, so one that cannot do 16 kHz still opens.
    sample_rate = int(sounddevice.query_devices(index)["default_samplerate"])
    wanted_bytes = int(PROBE_SECONDS * sample_rate) * 2
    blocks = queue.Queue()
    collected = bytearray()
    with sounddevice.RawInputStream(
        samplerate=sample_rate, blocksize=0, dtype="int16", channels=1, device=index,
        callback=lambda indata, _frames, _time, _status: blocks.put(bytes(indata)),
    ):
        while len(collected) < wanted_bytes:
            try:
                collected += blocks.get(timeout=1.0)
            except queue.Empty:
                break
    return loudness(bytes(collected))
