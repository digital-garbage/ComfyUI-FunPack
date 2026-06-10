"""Unit tests for Pulse temporal style schedule math."""

import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ltx_enhancements", _root / "ltx_enhancements.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

pulse_mult_for_progress = _mod.pulse_mult_for_progress
PULSE_PEAK_MULT = _mod.PULSE_PEAK_MULT
PULSE_FLOOR_MULT = _mod.PULSE_FLOOR_MULT


def test_pulse_starts_at_peak():
    mult = pulse_mult_for_progress(0.0, segment_count=3)
    assert abs(mult - PULSE_PEAK_MULT) < 1e-6


def test_pulse_segment_boundaries_reset_toward_peak():
    # End of segment 0 (progress ~1/3) should be near floor; start of segment 1 resets toward peak.
    end_seg0 = pulse_mult_for_progress(1.0 / 3.0 - 1e-6, segment_count=3)
    start_seg1 = pulse_mult_for_progress(1.0 / 3.0, segment_count=3)
    assert end_seg0 > PULSE_PEAK_MULT
    assert abs(start_seg1 - PULSE_PEAK_MULT) < 1e-6


def test_pulse_eases_down_within_segment():
    start = pulse_mult_for_progress(0.0, segment_count=1)
    end = pulse_mult_for_progress(1.0, segment_count=1)
    mid = pulse_mult_for_progress(0.5, segment_count=1)
    assert start < mid < end
    assert abs(end - PULSE_FLOOR_MULT) < 1e-6


def test_pulse_three_segments():
    samples = [pulse_mult_for_progress(p, segment_count=3) for p in (0.0, 0.16, 0.33, 0.5, 0.66, 0.99)]
    # Each segment should show an upward jump at boundaries (reset to peak).
    assert samples[0] < samples[1] < samples[2]
    assert samples[2] > samples[3]
    assert samples[3] < samples[4]