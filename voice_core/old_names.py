from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from operator import attrgetter
from typing import Any


def answers_to_old_names(**renamed: str) -> Callable[[Any], Any]:
    def answering(named: Any) -> Any:
        if not isinstance(named, type):
            return _taking_old_names(named, renamed)
        named.__init__ = _taking_old_names(named.__init__, renamed)
        for old, new in renamed.items():
            setattr(named, old, property(attrgetter(new)))
        return named
    return answering


def _taking_old_names(call: Callable[..., Any], renamed: Mapping[str, str]) -> Callable[..., Any]:
    @wraps(call)
    def taking(*args: Any, **named: Any) -> Any:
        for old, new in renamed.items():
            if old in named:
                named[new] = named.pop(old)
        return call(*args, **named)
    return taking
