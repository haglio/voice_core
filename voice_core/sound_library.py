from __future__ import annotations

import ctypes
import importlib
import sys
from types import ModuleType

if sys.platform == "win32":
    import msvcrt

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.SetStdHandle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    _kernel32.SetStdHandle.restype = ctypes.c_int

_STD_ERROR_HANDLE = 0xFFFFFFF4


def load_sounddevice() -> ModuleType:
    sounddevice = importlib.import_module("sounddevice")
    _point_windows_standard_error_at_descriptor_2()
    return sounddevice


def _point_windows_standard_error_at_descriptor_2() -> None:
    if sys.platform != "win32":
        return
    try:
        descriptor_2 = msvcrt.get_osfhandle(2)
    except OSError:
        return
    _kernel32.SetStdHandle(_STD_ERROR_HANDLE, descriptor_2)
