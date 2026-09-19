from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol

from voice_core.listener import SECOND_OPINION_PATIENCE_S

# A live listener is back within one poll of its microphone plus whatever reading the second
# engine has under way; a wedged one is not something a closing window waits on.
STOP_PATIENCE_S = SECOND_OPINION_PATIENCE_S + 2.0


class _Listens(Protocol):
    def run(self) -> None: ...

    def stop(self) -> None: ...


class ListeningThread:
    def __init__(self, build: Callable[[], _Listens], *,
                 failed: Callable[[Exception], None],
                 patience_s: float = STOP_PATIENCE_S) -> None:
        self._build = build
        self._failed = failed
        self._patience_s = patience_s
        self._listener: _Listens | None = None
        self._thread: threading.Thread | None = None

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
        if thread is not None:
            thread.join(timeout=self._patience_s)
