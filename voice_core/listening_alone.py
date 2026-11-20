from __future__ import annotations

from typing import Protocol

from app_support.machine_line import (
    THE_NAMESPACE_EVERY_LOGIN_SESSION_SHARES,
    SingleInstanceLock,
    held_elsewhere,
)

THE_CLAIMS_NAME = f"{THE_NAMESPACE_EVERY_LOGIN_SESSION_SHARES}haglio_voice_listening_alone"


class ListeningAloneClaim(Protocol):
    def take(self) -> bool: ...

    def give_back(self) -> None: ...

    def held_elsewhere(self) -> bool: ...


class NobodysClaim:
    def take(self) -> bool:
        return False

    def give_back(self) -> None:
        pass

    def held_elsewhere(self) -> bool:
        return False


class MachineClaim:
    def __init__(self, lock: SingleInstanceLock) -> None:
        self._lock = lock

    @classmethod
    def on_this_computer(cls) -> MachineClaim:
        return cls(SingleInstanceLock(THE_CLAIMS_NAME))

    def take(self) -> bool:
        return self._lock.acquire(timeout=0)

    def give_back(self) -> None:
        self._lock.close()

    def held_elsewhere(self) -> bool:
        return held_elsewhere(self._lock)
