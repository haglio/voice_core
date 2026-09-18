"""Every family checkout this one runs is named at a version, not left to chance."""
from __future__ import annotations

from pathlib import Path

from app_support.dependencies import (
    assert_every_sibling_is_pinned,
    assert_the_version_comes_from_the_tag,
)

ROOT = Path(__file__).resolve().parent.parent


def test_every_sibling_is_pinned():
    assert_every_sibling_is_pinned(ROOT / "pyproject.toml")


def test_the_version_comes_from_the_tag():
    assert_the_version_comes_from_the_tag(ROOT / "pyproject.toml")
