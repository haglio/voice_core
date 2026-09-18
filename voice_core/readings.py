from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

UNKNOWN = "[unk]"


def build_grammar(phrases: Iterable[str]) -> str:
    return json.dumps([*sorted(phrases), UNKNOWN])


def _parsed(raw_json: str) -> dict:
    try:
        parsed = json.loads(raw_json) if raw_json else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def partial_text(raw_json: str) -> str:
    text = str(_parsed(raw_json).get("partial", "")).strip()
    return "" if text == UNKNOWN else text


@dataclass(frozen=True)
class Hypothesis:
    text: str
    # Empty when vosk ranks whole readings: unscored, never zero.
    confidences: tuple[float, ...] = ()


def hypotheses(raw_json: str) -> list[Hypothesis]:
    data = _parsed(raw_json)
    found = []
    for reading in data.get("alternatives") or [data]:
        text = reading.get("text", "").strip()
        if text:
            words = reading.get("result") or []
            found.append(Hypothesis(text, tuple(w["conf"] for w in words if "conf" in w)))
    return found
