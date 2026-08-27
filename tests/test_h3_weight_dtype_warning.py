"""MiniMax H3 at a precision that breaks it should say so where you can see it.

The loader's notes go into a status string the Editor's fixed graph never renders, so a
weight_dtype that corrupts the audio was invisible until the audio came back corrupted.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

comfy = pytest.importorskip("comfy")
import loaders  # noqa: E402
import funpack_log as L  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    L.reset()
    yield
    L.reset()


@pytest.fixture
def h3(monkeypatch):
    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_model", lambda _model: True)


@pytest.fixture
def not_h3(monkeypatch):
    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_model", lambda _model: False)


def test_fp16_on_h3_is_called_out(h3, capsys):
    loaders._warn_h3_weight_dtype(object(), "fp16")
    assert "wants bf16 weights, not fp16" in capsys.readouterr().out


def test_the_fix_survives_the_console_trim(h3, capsys):
    """The renderer keeps the first sentence and trims the rest. A message that explains
    before it answers arrives with the answer cut off."""
    loaders._warn_h3_weight_dtype(object(), "fp16")
    assert "bf16" in capsys.readouterr().out.splitlines()[0]


def test_the_reason_is_there_in_full_under_verbose(h3, capsys):
    L.set_verbose(True)
    try:
        loaders._warn_h3_weight_dtype(object(), "fp16")
        assert "soundtrack breaks first" in capsys.readouterr().out
    finally:
        L.set_verbose(False)


def test_the_measured_fp8_variant_is_called_out(h3, capsys):
    loaders._warn_h3_weight_dtype(object(), "fp8_e4m3fn_fast")
    assert "not fp8_e4m3fn_fast" in capsys.readouterr().out


def test_bf16_says_nothing(h3, capsys):
    loaders._warn_h3_weight_dtype(object(), "bf16")
    assert capsys.readouterr().out == ""


def test_only_measured_dtypes_are_named():
    """Guessing at the rest would be worse than silence — a warning nobody can act on
    teaches people to ignore the ones that matter."""
    assert set(loaders.H3_RISKY_WEIGHT_DTYPES) == {"fp16", "fp8_e4m3fn_fast"}
    assert all(d in loaders.WEIGHT_DTYPES for d in loaders.H3_RISKY_WEIGHT_DTYPES)


def test_another_family_at_fp16_is_left_alone(not_h3, capsys):
    """fp16 is a perfectly ordinary choice on LTX. This is an H3 fact, not a global one."""
    loaders._warn_h3_weight_dtype(object(), "fp16")
    assert capsys.readouterr().out == ""


def test_it_is_said_once_not_every_load(h3, capsys):
    for _ in range(4):
        loaders._warn_h3_weight_dtype(object(), "fp16")
    assert capsys.readouterr().out.count("wants bf16") == 1


def test_a_model_it_cannot_identify_is_not_guessed_at(monkeypatch, capsys):
    import minimax_h3

    def boom(_model):
        raise RuntimeError("unreadable")
    monkeypatch.setattr(minimax_h3, "is_h3_model", boom)
    loaders._warn_h3_weight_dtype(object(), "fp16")
    assert capsys.readouterr().out == ""


def test_the_loader_actually_calls_it():
    import inspect
    assert "_warn_h3_weight_dtype(model, weight_dtype)" in \
        inspect.getsource(loaders.FunPackDiffusionModelLoader.load_model)
