from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from typing import Any


class LoadedModels:
    def __init__(self) -> None:
        self._models: dict[Hashable, Any] = {}
        self._locks: dict[Hashable, threading.Lock] = {}
        self._guard = threading.Lock()

    def take(self, name: Hashable, load: Callable[[], Any]) -> Any:
        with self._guard:
            lock = self._locks.setdefault(name, threading.Lock())
        with lock:
            if name not in self._models:
                self._models[name] = load()
            return self._models[name]


IN_THIS_PROCESS = LoadedModels()
