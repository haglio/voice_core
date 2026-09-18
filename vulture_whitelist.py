"""Vulture whitelist -- false positives that are not dead code.

This is a library: what it offers is read by the apps, not by anything here.
vulture matches by bare name, so tests/test_dead_code.py asserts every entry
still answers a report, and an entry may only be added with the app that reads it.
"""
from __future__ import annotations
