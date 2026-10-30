from __future__ import annotations

import ctypes
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

from voice_core import sound_library
from voice_core.sound_library import load_sounddevice

STD_ERROR_HANDLE = 0xFFFFFFF4


class _SetStdHandle:
    def __init__(self):
        self.pointed = []

    def __call__(self, which, handle):
        self.pointed.append((which, handle))
        return 1


def _on_windows(monkeypatch, *, handle_of):
    set_std_handle = _SetStdHandle()
    monkeypatch.setattr(ctypes, "WinDLL", lambda name: SimpleNamespace(SetStdHandle=set_std_handle),
                        raising=False)
    monkeypatch.setattr(sound_library, "msvcrt", SimpleNamespace(get_osfhandle=handle_of),
                        raising=False)
    return set_std_handle.pointed


def _a_sounddevice(monkeypatch):
    sounddevice = ModuleType("sounddevice")
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    return sounddevice


def test_sounddevice_comes_back_with_windows_error_output_pointed_at_descriptor_2(monkeypatch):
    sounddevice = _a_sounddevice(monkeypatch)
    pointed = _on_windows(monkeypatch, handle_of=lambda descriptor: 1000 + descriptor)

    assert load_sounddevice() is sounddevice
    assert pointed == [(STD_ERROR_HANDLE, 1002)]


def test_error_output_is_left_alone_where_descriptor_2_has_no_windows_handle(monkeypatch):
    sounddevice = _a_sounddevice(monkeypatch)
    pointed = _on_windows(monkeypatch, handle_of=Mock(side_effect=OSError(9, "Bad file descriptor")))

    assert load_sounddevice() is sounddevice
    assert pointed == []


def test_sounddevice_loads_where_ctypes_carries_no_windows_half(monkeypatch):
    sounddevice = _a_sounddevice(monkeypatch)
    asked = Mock()
    monkeypatch.setattr(sound_library, "msvcrt", SimpleNamespace(get_osfhandle=asked), raising=False)
    monkeypatch.delattr(ctypes, "WinDLL", raising=False)

    assert load_sounddevice() is sounddevice
    asked.assert_not_called()


def test_a_windowed_app_s_error_output_is_still_what_its_programs_inherit_once_sounddevice_loads(
        tmp_path):
    loads_voice_then_starts_a_program = tmp_path / "loads_voice_then_starts_a_program.py"
    loads_voice_then_starts_a_program.write_text(textwrap.dedent("""
        import ctypes
        import msvcrt
        import subprocess
        import sys

        from voice_core.sound_library import load_sounddevice

        load_sounddevice()
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        answers = [str(kernel32.GetStdHandle(-12) == msvcrt.get_osfhandle(2))]
        try:
            subprocess.check_output([sys.executable, "-c", "pass"],
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            answers.append("started")
        except OSError as refused:
            answers.append(repr(refused))
        with open(sys.argv[1], "w", encoding="utf-8") as told:
            told.write(" ".join(answers))
    """), encoding="utf-8")
    answer = tmp_path / "answer.txt"
    windowed_python = Path(sys.executable).with_name("pythonw.exe")
    with (tmp_path / "its_error_output.txt").open("wb") as its_error_output:
        subprocess.run([str(windowed_python), str(loads_voice_then_starts_a_program), str(answer)],
                       stdin=subprocess.DEVNULL, stdout=its_error_output, stderr=its_error_output,
                       creationflags=subprocess.CREATE_NO_WINDOW, check=True, timeout=60)

    assert answer.read_text(encoding="utf-8") == "True started"


def test_importing_voice_binds_nothing_where_ctypes_carries_no_windows_half():
    strips_the_windows_half_then_imports_voice = textwrap.dedent("""
        import ctypes
        for name in ("windll", "oledll", "WinDLL", "OleDLL", "WINFUNCTYPE", "HRESULT",
                     "get_last_error", "set_last_error"):
            if hasattr(ctypes, name):
                delattr(ctypes, name)
        import voice_core.listener
    """)
    result = subprocess.run([sys.executable, "-c", strips_the_windows_half_then_imports_voice],
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
