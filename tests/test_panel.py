import ctypes

import pytest

from tools.common import WINDOW_US, native


class Panel(ctypes.Structure):
    _fields_ = [("awaiting_window", ctypes.c_bool), ("last_window_us", ctypes.c_int64)]


@pytest.fixture
def panel():
    lib = native()
    lib.panel_init.argtypes = [ctypes.POINTER(Panel), ctypes.c_int64]
    lib.panel_init.restype = None
    lib.panel_note_window.argtypes = [ctypes.POINTER(Panel), ctypes.c_int64]
    lib.panel_note_window.restype = None
    for name in ["panel_audio_timeout", "panel_red_led"]:
        function = getattr(lib, name)
        function.argtypes = [ctypes.POINTER(Panel), ctypes.c_int64]
        function.restype = ctypes.c_bool
    state = Panel()
    lib.panel_init(ctypes.byref(state), 0)
    return lib, ctypes.byref(state)


def test_initial_grace_waits_for_first_window_but_reports_stalled_capture(panel):
    lib, pointer = panel
    deadline = 2 * WINDOW_US + 1_500_000
    assert not lib.panel_audio_timeout(pointer, deadline - 1)
    assert not lib.panel_red_led(pointer, deadline - 1)
    assert lib.panel_audio_timeout(pointer, deadline)


def test_red_led_blinks_after_capture_stops_and_recovers_with_fresh_audio(panel):
    lib, pointer = panel
    lib.panel_note_window(pointer, 0)
    deadline = WINDOW_US + 1_500_000
    assert not lib.panel_audio_timeout(pointer, deadline - 1)
    assert lib.panel_audio_timeout(pointer, deadline)
    assert not lib.panel_red_led(pointer, 4_000_000)
    assert lib.panel_red_led(pointer, 4_250_000)
    lib.panel_note_window(pointer, 4_300_000)
    assert not lib.panel_audio_timeout(pointer, 4_300_000)
    assert not lib.panel_red_led(pointer, 4_300_000)


def test_regular_audio_windows_keep_status_led_off(panel):
    lib, pointer = panel
    for index in range(1, 20):
        now = index * WINDOW_US
        assert not lib.panel_audio_timeout(pointer, now)
        lib.panel_note_window(pointer, now)
        assert not lib.panel_red_led(pointer, now)
