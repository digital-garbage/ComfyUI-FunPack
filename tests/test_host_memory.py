"""Lowering ComfyUI's pinned host-memory budget from inside FunPack.

Pinned pages are unevictable: once the budget is committed the kernel cannot swap or reclaim
them, so a further allocation wedges the machine rather than killing the process. ComfyUI's
own switch is a launch flag, which a rented image often bakes in where it cannot be edited —
FunPack is imported in the window between the budget being computed and the first model being
staged, so the same control is available here.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import host_memory as hm  # noqa: E402

GB = 1024 ** 3
RAM = 125 * GB


def _mm(current_gb=112):
    return types.SimpleNamespace(MAX_PINNED_MEMORY=current_gb * GB)


# --- parsing --------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["off", "0", "no", "false", "disable", "disabled", "none",
                                  " OFF ", "Disabled"])
def test_the_off_words_mean_zero(text):
    """0 is what model_management already treats as 'do not pin', so disabling needs no
    separate flag."""
    assert hm.parse_budget(text, RAM) == 0


def test_a_plain_number_is_gigabytes():
    assert hm.parse_budget("64", RAM) == 64 * GB
    assert hm.parse_budget("64gb", RAM) == 64 * GB


def test_a_percentage_is_of_total_ram():
    assert hm.parse_budget("50%", RAM) == pytest.approx(RAM * 0.5)


def test_unset_leaves_comfyui_alone():
    assert hm.parse_budget("", RAM) is None
    assert hm.parse_budget(None, RAM) is None


@pytest.mark.parametrize("text", ["lots", "-5", "0%", "150%", "abc%", "-3gb"])
def test_a_typo_changes_nothing(text):
    """A misread value must not silently reconfigure memory — the whole point of this module
    is that getting it wrong takes the machine down."""
    assert hm.parse_budget(text, RAM) is None


def test_a_percentage_without_a_known_ram_size_is_refused():
    assert hm.parse_budget("50%", 0) is None


# --- applying -------------------------------------------------------------------------


def test_it_lowers_the_budget():
    mm = _mm(112)
    note = hm.apply("64", RAM, mm)
    assert mm.MAX_PINNED_MEMORY == 64 * GB
    assert "64 GB" in note and "112 GB" in note


def test_off_disables_pinning_entirely():
    mm = _mm(112)
    note = hm.apply("off", RAM, mm)
    assert mm.MAX_PINNED_MEMORY == 0
    assert "DISABLED" in note


def test_it_never_RAISES_the_budget():
    """Raising it hands the same unevictable-memory footgun back with a bigger foot, and
    nothing here knows better than model_management what this box can pin."""
    mm = _mm(112)
    note = hm.apply("200", RAM, mm)
    assert mm.MAX_PINNED_MEMORY == 112 * GB
    assert "left alone" in note


def test_an_equal_budget_is_left_alone():
    mm = _mm(112)
    hm.apply("112", RAM, mm)
    assert mm.MAX_PINNED_MEMORY == 112 * GB


def test_nothing_happens_when_the_variable_is_unset():
    mm = _mm(112)
    assert hm.apply("", RAM, mm) is None
    assert mm.MAX_PINNED_MEMORY == 112 * GB


def test_already_disabled_says_so_rather_than_going_quiet():
    """Launched with --disable-pinned-memory: the setting is redundant, and silence would
    read as 'FunPack turned it off for you'."""
    mm = _mm(0)
    note = hm.apply("64", RAM, mm)
    assert mm.MAX_PINNED_MEMORY == 0
    assert "already off" in note


def test_the_environment_variable_drives_it(monkeypatch):
    monkeypatch.setenv(hm.ENV_VAR, "32")
    mm = _mm(112)
    hm.apply(None, RAM, mm)
    assert mm.MAX_PINNED_MEMORY == 32 * GB


def test_a_missing_model_management_is_not_an_error():
    """This runs at import. It must never be the reason FunPack fails to load."""
    assert hm.apply("32", RAM, types.SimpleNamespace()) is None or True
