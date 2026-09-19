from __future__ import annotations

import threading

from voice_core.listening_thread import ListeningThread

PATIENCE_S = 10.0


class _Listener:
    """Listens until told to stop, and says when it began and when it was done."""

    def __init__(self, *, fails_with=None, never_comes_back=False):
        self.began = threading.Event()
        self.done = threading.Event()
        self._told_to_stop = threading.Event()
        self._fails_with = fails_with
        self._never_comes_back = never_comes_back
        self.let_go = threading.Event()

    def run(self):
        self.began.set()
        if self._fails_with is not None:
            raise self._fails_with
        if self._never_comes_back:
            self.let_go.wait(PATIENCE_S)
        else:
            assert self._told_to_stop.wait(PATIENCE_S), "nobody stopped the listener"
        self.done.set()

    def stop(self):
        self._told_to_stop.set()


def _never(exc):
    raise AssertionError(f"listening failed: {exc!r}")


def test_stopping_returns_only_once_the_listener_is_done():
    listener = _Listener()
    thread = ListeningThread(lambda: listener, failed=_never)

    thread.start()
    assert listener.began.wait(PATIENCE_S)
    thread.stop()

    assert listener.done.is_set()


def test_starting_twice_runs_one_listener():
    built = []

    def build():
        built.append(_Listener())
        return built[-1]

    thread = ListeningThread(build, failed=_never)
    thread.start()
    thread.start()
    thread.stop()

    assert len(built) == 1


def test_once_stopped_it_can_be_started_again_with_a_listener_of_its_own():
    built = []

    def build():
        built.append(_Listener())
        return built[-1]

    thread = ListeningThread(build, failed=_never)
    thread.start()
    thread.stop()
    thread.start()
    assert built[-1].began.wait(PATIENCE_S)
    thread.stop()

    assert [listener.done.is_set() for listener in built] == [True, True]


def test_stopping_one_that_never_started_is_nothing():
    ListeningThread(_Listener, failed=_never).stop()


def test_a_listener_that_never_comes_back_does_not_hold_whoever_is_stopping_it():
    wedged = _Listener(never_comes_back=True)
    thread = ListeningThread(lambda: wedged, failed=_never, patience_s=0.05)
    stopped = threading.Event()

    thread.start()
    assert wedged.began.wait(PATIENCE_S)
    threading.Thread(target=lambda: (thread.stop(), stopped.set()), daemon=True).start()
    try:
        assert stopped.wait(PATIENCE_S / 2), "stop() is still waiting"
    finally:
        wedged.let_go.set()


def test_what_ended_the_listening_is_handed_to_the_app():
    ended_by = []
    told = threading.Event()

    def failed(exc):
        ended_by.append(exc)
        told.set()

    gone = OSError("no input device")
    thread = ListeningThread(lambda: _Listener(fails_with=gone), failed=failed)

    thread.start()
    assert told.wait(PATIENCE_S)
    thread.stop()

    assert ended_by == [gone]
