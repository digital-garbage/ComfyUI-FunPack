"""H3 attention temperature (_install_h3_attn_temperature): a manual randomizer, NOT tied
to ratings. Flattens the softmax at named blocks by scaling Q before the dot product.
Exercised at the install/override level with fake tensors -- no real attention backend."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, ".")
import _comfy_stubs  # noqa: E402, F401
from samplers import FunPackLTXAVSceneChainSampler as S  # noqa: E402


class _FakeModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        m = _FakeModel()
        m.model_options = dict(self.model_options)
        return m


def _run_block(patched, block, seq_len=4):
    """Fires block N's hook and, while "inside" it, calls the installed attention override
    -- the same two-step real generations go through (hook sets a flag, then the block's
    own Attention.forward triggers the override)."""
    to = patched.model_options["transformer_options"]
    dit = to["patches_replace"]["dit"]
    hook = dit[("double_block", block)]
    override = to["optimized_attention_override"]
    calls = []

    def fake_func(q, k, v, heads, mask=None, skip_reshape=True, **kw):
        calls.append({"q": q})
        return torch.zeros(q.shape[0], q.shape[-2], heads * q.shape[-1])

    def original_block(args):
        override(fake_func, args["q"], args["k"], args["v"], 1, skip_reshape=True)
        return {"img": args["img"]}

    hook({"img": torch.ones(seq_len, 3),
         "q": torch.ones(1, 1, seq_len, 2), "k": torch.ones(1, 1, seq_len, 2),
         "v": torch.ones(1, 1, seq_len, 2)},
        {"original_block": original_block})
    return calls[0]["q"] if calls else None


def test_zero_strength_is_a_true_noop():
    model = _FakeModel()
    node = S()
    assert node._install_h3_attn_temperature(model, 0.0, "40-49") is model
    assert node._install_h3_attn_temperature(model, "", "40-49") is model


def test_a_positive_strength_with_no_blocks_is_also_a_noop():
    model = _FakeModel()
    node = S()
    assert node._install_h3_attn_temperature(model, 0.5, "") is model


def test_scales_q_only_inside_a_named_block():
    node = S()
    patched = node._install_h3_attn_temperature(_FakeModel(), 1.0, "5")  # temperature = 2.0
    q_at_5 = _run_block(patched, 5)
    assert torch.allclose(q_at_5, torch.full((1, 1, 4, 2), 0.5)), "Q should be halved (1/temp)"


def test_no_hook_installed_for_an_unnamed_block():
    node = S()
    patched = node._install_h3_attn_temperature(_FakeModel(), 1.0, "5")
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    assert ("double_block", 9) not in dit


def test_calling_the_override_outside_any_named_block_is_a_passthrough():
    """The override is model-wide (comfy has one slot for it), so it must recognize when
    the CURRENT attention call isn't one of the flattened blocks and leave Q untouched --
    e.g. a call that happens between block hooks, or a block never wrapped."""
    node = S()
    patched = node._install_h3_attn_temperature(_FakeModel(), 1.0, "5")
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    calls = []

    def fake_func(q, k, v, heads, mask=None, skip_reshape=True, **kw):
        calls.append(q)
        return torch.zeros(1, 4, 2)

    q = torch.ones(1, 1, 4, 2)
    override(fake_func, q, q, q, 1, skip_reshape=True)
    assert torch.allclose(calls[0], q), "no active block -> Q must be untouched"


def test_chains_through_an_already_installed_attention_override():
    """If h3_av_decouple (or anything else) already claimed optimized_attention_override,
    installing the temperature knob must chain through it, not replace it silently."""
    inner_calls = []

    def inner_override(func, q, k, v, heads, mask=None, skip_reshape=True, **kw):
        inner_calls.append(mask)
        return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kw)

    model = _FakeModel()
    model.model_options["transformer_options"] = {"optimized_attention_override": inner_override}
    node = S()
    patched = node._install_h3_attn_temperature(model, 1.0, "5")
    _run_block(patched, 5)
    assert len(inner_calls) == 1, "the previously-installed override must still be reached"


if __name__ == "__main__":
    test_zero_strength_is_a_true_noop()
    test_a_positive_strength_with_no_blocks_is_also_a_noop()
    print("ok (run via pytest for the rest)")
