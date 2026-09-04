"""REINS steer_block went from a single combo choice to free text ("31-40", "4,5,6"),
each named block steering with its OWN learned direction. Exercised at the install level:
does the resulting hook actually inject at every named block and NOWHERE else."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, ".")
import _comfy_stubs  # noqa: E402, F401
from samplers import FunPackLTXAVSceneChainSampler as S  # noqa: E402
import h3_repr_steering as rs  # noqa: E402


class _FakeModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        m = _FakeModel()
        m.model_options = dict(self.model_options)
        return m


def _run_block(patched, block, seq_len=4):
    """Invoke the installed hook for `block` and return the resulting "img" tensor."""
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    hook = dit[("double_block", block)]
    img = torch.ones(seq_len, 3)
    args = {"img": img, "mod_segments": [(0, seq_len, 0)]}  # all rows tagged video
    extra = {"original_block": lambda a: {"img": a["img"]}}
    return hook(args, extra)["img"]


def test_multiple_named_blocks_each_steer_with_their_own_direction(monkeypatch):
    directions = {5: torch.tensor([1.0, 0.0, 0.0]), 10: torch.tensor([0.0, 1.0, 0.0])}

    def fake_direction(_key, block=None):
        d = directions.get(block)
        return (d, 5, 5) if d is not None else (None, 0, 0)

    monkeypatch.setattr(rs, "direction", fake_direction)
    node = S()
    patched = node._install_h3_repr_steering(
        _FakeModel(), "key", strength=1.0, capture_holder=[{}], steer_block="5,10")

    out5 = _run_block(patched, 5)
    out10 = _run_block(patched, 10)
    out6 = _run_block(patched, 6)  # captured, never steered

    assert not torch.allclose(out5, torch.ones(4, 3)), "block 5 should have been steered"
    assert not torch.allclose(out10, torch.ones(4, 3)), "block 10 should have been steered"
    assert torch.allclose(out6, torch.ones(4, 3)), "an unnamed block must stay untouched"
    # Different learned directions -> different injected deltas.
    assert not torch.allclose(out5, out10)


def test_a_range_steers_every_block_in_it(monkeypatch):
    monkeypatch.setattr(rs, "direction", lambda _k, block=None: (torch.tensor([1.0, 0.0, 0.0]), 5, 5))
    node = S()
    patched = node._install_h3_repr_steering(
        _FakeModel(), "key", strength=1.0, capture_holder=[{}], steer_block="7-9")

    for b in (7, 8, 9):
        assert not torch.allclose(_run_block(patched, b), torch.ones(4, 3)), b
    assert torch.allclose(_run_block(patched, 6), torch.ones(4, 3))
    assert torch.allclose(_run_block(patched, 10), torch.ones(4, 3))


def test_a_block_without_enough_rated_data_only_captures(monkeypatch):
    monkeypatch.setattr(rs, "direction", lambda _k, block=None: (None, 1, 0))
    node = S()
    capture_holder = [{}]
    patched = node._install_h3_repr_steering(
        _FakeModel(), "key", strength=1.0, capture_holder=capture_holder, steer_block="5")

    out = _run_block(patched, 5)
    assert torch.allclose(out, torch.ones(4, 3)), "no direction yet -- must not inject"
    assert 5 in capture_holder[0], "capture should still happen regardless of steering"


def test_empty_steer_block_falls_back_to_default_block():
    node = S()
    patched = node._install_h3_repr_steering(
        _FakeModel(), "", strength=1.0, capture_holder=[{}], steer_block="")
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    assert ("double_block", rs.DEFAULT_BLOCK) in dit


if __name__ == "__main__":
    test_empty_steer_block_falls_back_to_default_block()
    print("ok (run via pytest for the monkeypatch-dependent cases)")
