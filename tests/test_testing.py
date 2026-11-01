from __future__ import annotations

from voice_core.commands import CommandRules
from voice_core.listener import CommandListener, ListenerEvents, ListenerSettings
from voice_core.pauses import PauseSettings
from voice_core.testing import said, scripted_engines

RULES = CommandRules(phrases=frozenset({"side beta"}), never_repaired=lambda phrase: False)
SETTINGS = ListenerSettings(model_name="a-model", device_name="desk", poll_seconds=0.001)


def _run(utterances, *, until=None):
    engines = scripted_engines(utterances)
    heard, forming = [], []

    def on_heard(one):
        heard.append(one.recognition.phrase)
        if one.recognition.phrase == until:
            listener.stop()

    listener = CommandListener(RULES, SETTINGS, ListenerEvents(
        heard=on_heard, partial=forming.append), engines)
    listener.run()
    return heard, forming, engines.sounddevice


def test_a_scripted_utterance_is_heard_as_its_reading_from_a_microphone_that_opens_and_closes():
    heard, _, microphone = _run([said("side beta")], until="side beta")

    assert heard == ["side beta"]
    assert (len(microphone.opened), microphone.closed) == (1, 1)


def test_the_words_forming_are_offered_as_they_change_and_cleared_before_the_reading():
    words = ["side"] * PauseSettings().min_speech_frames + ["side beta"]
    _, forming, _ = _run([said("side beta", forming=words)], until="side beta")

    assert forming == ["side", "side beta", ""]
