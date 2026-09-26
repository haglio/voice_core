# voice_core

How every Haglio app hears. One listener, shared, so a fix to hearing is made once:
Fun Time, Evolver's backfill tool and Origenerator each used to carry their own.

An app hands over a list of phrases and a few callables; what a phrase *means* stays
with the app.

```python
from voice_core.commands import CommandRules
from voice_core.listener import CommandListener, Engines, ListenerEvents, ListenerSettings
from voice_core.listening_thread import ListeningThread
from voice_core.whisper_reader import WhisperReader

listening = ListeningThread(lambda: CommandListener(
    CommandRules(phrases=frozenset({"next", "left next", "quit"}),
                 never_rescued={"quit"}.__contains__),
    ListenerSettings(model_name="vosk-model-en-us-0.22-lgraph", device_name="Brio",
                     miss_dir=state_dir / "voice_misses"),
    ListenerEvents(heard=on_heard),
    Engines(second_opinion=WhisperReader()),
), failed=on_failure)
listening.start()
```

`ListeningThread` runs a listener on a thread of its own and can be stopped and started
again, each start with a fresh listener; `stop()` waits for the microphone to close, though
never on a listener that has wedged. `on_failure` receives what ended the listening: a
`RecognizerUnavailable`, a `MicrophoneUnavailable`, or whatever else was raised.

`on_heard` receives a `Heard`: the `Recognition` (the phrase that was said, or which
kind of miss it was), when the utterance began, how loud it was, and its audio.

## Two engines, because neither is good alone

Measured on recordings of the owner's voice (see CLAUDE.md for where they are and
why nothing else counts):

| | missed commands recovered | ordinary talk heard as a command |
|---|---|---|
| vosk mid-size, grammar + guards | 6 of 10 | 8.6% |
| whisper, prompted with command words | 9 of 10 | 17.2% |
| vosk proposes, whisper (base) must agree | 7 of 10 | 0.3% |

Vosk takes a phrase list of any length and reads a cough as a one-word command.
Whisper's prompt holds 223 tokens and drags ordinary speech onto whatever it names.
So vosk proposes -- every phrase among its five ranked readings is a candidate -- and
whisper, prompted with just that utterance's candidates and handed only the stretch of it
vosk heard them in, picks one or refuses them all; before refusing, it reads the stretch
again lifted to full level, where it may not call anything silence
(`second_opinion.settle`). Over 1,196 utterances of the owner's ordinary dictation that let
14 through where one reading of the whole utterance let 12, and it read the command in 4 of
11 kept clips of commands the one reading had doubted, against 2.
`CommandRules.stands_alone` names the phrases an app wants acted on without that
half-second wait.

Without `Engines.second_opinion` the listener is vosk alone, with the guards Fun Time
grew: a lower-ranked reading is taken only when it shares a word with the first and the
app has not ruled the phrase out of repairs, and an utterance quieter than
`SILENT_UTTERANCE_PEAK` is set aside.

## Installing

An app names a tag of this repo in its `[project.dependencies]`:

    "voice-core @ git+https://github.com/haglio/voice_core@v0.1.N"

and asks for the `whisper` extra (or `faster-whisper` itself) where it wants the
second engine. The vosk model is downloaded by vosk on first use; whisper's by
faster-whisper.
