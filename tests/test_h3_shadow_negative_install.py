"""_install_h3_shadow_negative: install/no-op behaviour at the model-patching level.

Exercised with fake model/diffusion-model stand-ins -- no real H3 weights or GPU needed.
The per-block math itself (nag_blend, presentation-span detection, ...) is covered in
tests/test_h3_shadow_negative.py; this file is about whether the installer clones the
right model, hooks the right blocks, and is honest about not composing with the other
dit-patch mechanisms.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, ".")
import _comfy_stubs  # noqa: E402, F401
from samplers import FunPackLTXAVSceneChainSampler as S  # noqa: E402


class _FakeDiffusionModel:
    def __init__(self, n_blocks=4):
        self.blocks = [object() for _ in range(n_blocks)]
        self.hidden_size = 16


class _FakeModel:
    def __init__(self, dm=None):
        self.model_options = {}
        self._dm = dm if dm is not None else _FakeDiffusionModel()

    def clone(self):
        m = _FakeModel(self._dm)
        m.model_options = dict(self.model_options)
        return m

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self._dm
        raise KeyError(name)


def _negative(tensor=None):
    return [(tensor if tensor is not None else torch.randn(1, 3, 16), {})]


def test_both_scales_at_one_is_a_true_noop():
    model = _FakeModel()
    node = S()
    out = node._install_h3_shadow_negative(model, _negative(), 1.0, 1.0, 2.5, 0.35, 0.0, 0.6)
    assert out is model


def test_unparseable_scale_is_a_noop():
    model = _FakeModel()
    node = S()
    out = node._install_h3_shadow_negative(model, _negative(), "nope", 1.0, 2.5, 0.35, 0.0, 0.6)
    assert out is model


def test_nan_scale_is_a_noop_not_a_silent_corruption(capsys):
    """NaN fails `video_scale == 1.0`, so that alone would let it through and NAG-blend
    every row to NaN with no warning -- must be caught explicitly, same as the codebase's
    other dit-patch strength guards (av_decouple, q_steering)."""
    model = _FakeModel()
    node = S()
    out = node._install_h3_shadow_negative(model, _negative(), float("nan"), 1.0,
                                           2.5, 0.35, 0.0, 0.6)
    assert out is model
    assert "NaN or infinite" in capsys.readouterr().out


def test_inf_tau_is_a_noop():
    model = _FakeModel()
    node = S()
    out = node._install_h3_shadow_negative(model, _negative(), 3.0, 1.0,
                                           float("inf"), 0.35, 0.0, 0.6)
    assert out is model


def test_nan_percent_is_a_noop():
    model = _FakeModel()
    node = S()
    out = node._install_h3_shadow_negative(model, _negative(), 3.0, 1.0,
                                           2.5, 0.35, float("nan"), 0.6)
    assert out is model


def test_missing_negative_is_a_noop():
    model = _FakeModel()
    node = S()
    out = node._install_h3_shadow_negative(model, [], 3.0, 1.0, 2.5, 0.35, 0.0, 0.6)
    assert out is model


def test_negative_with_empty_tensor_is_a_noop():
    model = _FakeModel()
    node = S()
    out = node._install_h3_shadow_negative(model, _negative(torch.zeros(1, 0, 16)),
                                           3.0, 1.0, 2.5, 0.35, 0.0, 0.6)
    assert out is model


def test_a_model_with_no_double_blocks_is_a_noop():
    class _NoBlocks:
        pass
    model = _FakeModel(dm=_NoBlocks())
    node = S()
    out = node._install_h3_shadow_negative(model, _negative(), 3.0, 1.0, 2.5, 0.35, 0.0, 0.6)
    assert out is model


def test_enabled_clones_and_hooks_every_block():
    model = _FakeModel(dm=_FakeDiffusionModel(n_blocks=5))
    node = S()
    patched = node._install_h3_shadow_negative(model, _negative(), 3.0, 1.0, 2.5, 0.35, 0.0, 0.6)
    assert patched is not model
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    assert {k for k in dit if k[0] == "double_block"} == {("double_block", i) for i in range(5)}


def test_swapped_start_and_end_percent_are_normalized():
    """A user typing 0.6/0.0 instead of 0.0/0.6 should not silently produce an empty
    (start > end) active window -- the installer should sort them."""
    import h3_shadow_negative as sn
    model = _FakeModel()
    node = S()
    patched = node._install_h3_shadow_negative(model, _negative(), 3.0, 1.0, 2.5, 0.35, 0.6, 0.0)
    hook = patched.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    state = next(c.cell_contents for c, name in
                zip(hook.__closure__, hook.__code__.co_freevars)
                if name == "state" and isinstance(c.cell_contents, sn.ShadowState))
    assert state.start_percent == 0.0 and state.end_percent == 0.6


def test_supersedes_an_existing_block_patch_rather_than_composing(capsys):
    """Non-composing by explicit design: it must fully replace whatever dit-patch mechanism
    (REINS/Q-steer/av_decouple/block-repeat/attn_temperature) already hooked a block, and
    say so loudly rather than silently overwrite it."""
    model = _FakeModel(dm=_FakeDiffusionModel(n_blocks=2))
    model.model_options["transformer_options"] = {"patches_replace": {"dit": {
        ("double_block", 0): lambda args, extra: {"img": args["img"] * 0.0},
    }}}
    node = S()
    patched = node._install_h3_shadow_negative(model, _negative(), 3.0, 1.0, 2.5, 0.35, 0.0, 0.6)
    out = capsys.readouterr().out
    assert "supersedes" in out
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    # The old hook must be GONE, not chained through -- calling the new one must not
    # reduce to zero the way the old one did.
    img = torch.ones(3, 4)
    result = dit[("double_block", 0)]({"img": img, "mod_segments": []},
                                      {"original_block": lambda a: {"img": a["img"]}})
    assert torch.allclose(result["img"], img), "should fall through to original_block, not the old hook"


def test_a_hooked_block_with_no_mod_segments_falls_through_untouched():
    model = _FakeModel(dm=_FakeDiffusionModel(n_blocks=1))
    node = S()
    patched = node._install_h3_shadow_negative(model, _negative(), 3.0, 1.0, 2.5, 0.35, 0.0, 0.6)
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    img = torch.ones(3, 4)
    out = dit[("double_block", 0)]({"img": img, "mod_segments": []},
                                   {"original_block": lambda a: {"img": a["img"] * 2.0}})
    assert torch.allclose(out["img"], img * 2.0)


if __name__ == "__main__":
    test_both_scales_at_one_is_a_true_noop()
    test_enabled_clones_and_hooks_every_block()
    print("ok (run via pytest for the rest)")
