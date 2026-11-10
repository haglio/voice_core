from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from voice_core.listener import READING_PATIENCE_S

_ROOM_FOR_A_MICROPHONE_POLL_S = 2.0
CLOSE_PATIENCE_S = READING_PATIENCE_S + _ROOM_FOR_A_MICROPHONE_POLL_S


class _Listens(Protocol):
    def run(self) -> None: ...

    def stop(self) -> None: ...


class ListeningThread:
    def __init__(self, build: Callable[[], _Listens], *,
                 failed: Callable[[Exception], None],
                 patience_s: float = CLOSE_PATIENCE_S) -> None:
        self._build = build
        self._failed = failed
        self._patience_s = patience_s
        self._listener: _Listens | None = None
        self._thread: threading.Thread | None = None
        self._closing: list[threading.Thread] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self._listener = self._build()
        self._thread = threading.Thread(target=self._listen, args=(self._listener,),
                                        daemon=True)
        self._thread.start()

    def _listen(self, listener: _Listens) -> None:
        try:
            listener.run()
        except Exception as exc:
            self._failed(exc)

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        thread, self._thread = self._thread, None
        if listener is not None:
            listener.stop()
        self._closing = [closing for closing in self._closing if closing.is_alive()]
        if thread is not None:
            self._closing.append(thread)

    def close(self) -> None:
        self.stop()
        deadline = time.monotonic() + self._patience_s
        for thread in self._closing:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._closing = []
