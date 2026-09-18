"""What this repo needs is what its pyproject says, in each of the ways it says it.

A package that imports something nobody declared works on the machine that
happened to have it and dies on the merge gate, which installs exactly what the
pyproject says. The gates are the family's (``app_support.dependencies``); what
is here is which packages are this repo's own and which trees to read.
"""
from __future__ import annotations

from pathlib import Path

from app_support.dependencies import (
    assert_every_dependency_is_bounded,
    assert_every_import_is_declared,
    assert_every_sibling_is_declared,
    assert_the_declared_floor_is_the_one_the_gate_runs,
)

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def test_every_third_party_import_is_declared():
    assert_every_import_is_declared(
        ROOT, [ROOT / "voice_core"], PYPROJECT, local=("voice_core",))


def test_every_requirement_has_an_upper_bound():
    assert_every_dependency_is_bounded(PYPROJECT)


def test_every_sibling_this_repo_needs_is_declared():
    assert_every_sibling_is_declared(
        ROOT, [ROOT / "voice_core", ROOT / "tests"], PYPROJECT)


def test_the_declared_floor_is_the_one_the_gate_runs():
    assert_the_declared_floor_is_the_one_the_gate_runs(
        PYPROJECT, ROOT / ".github" / "workflows" / "merge-gate.yml")
