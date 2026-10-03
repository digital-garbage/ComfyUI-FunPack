import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def test_strength_zero_changes_nothing_and_one_follows_the_memory():
    from modules.sampling.momentum import blend
    x, out = torch.randn(1, 4, 2, 4, 4), torch.randn(1, 4, 2, 4, 4)
    same, _ = blend(x, out, 0.5, None, 0.5, 0.0)
    assert torch.allclose(same, out, atol=1e-5)
    ema = torch.randn_like(x)
    full, _ = blend(x, out, 0.5, ema, 0.0, 1.0)            # decay 0: memory is this step's direction
    assert torch.allclose(full, out, atol=1e-5)
    mixed, new = blend(x, out, 0.5, ema, 0.5, 1.0)
    d = (x - out) / 0.5
    assert torch.allclose(new, 0.5 * ema + 0.5 * d, atol=1e-5)
    assert torch.allclose(mixed, x - 0.5 * new, atol=1e-5)


def _load(tiny, **v):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, status = FunPackLoadModifiers.execute(
        tiny.patcher, {"momentum": {"enabled": True, **v}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    return patched, wrap, status


def _to(i, steps=4):
    s = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": s, "sigmas": s[i:i + 1]}


def test_it_acts_below_the_threshold_and_carries_the_edit_into_the_next_step(tiny_h3):
    from conftest import packed_av, unpacked
    _p, wrap, status = _load(tiny_h3, strength=1.0, decay=0.5, below_sigma=0.8)
    torch.manual_seed(0)
    x0, shapes = packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))
    answers = [x0 * f for f in (0.9, 0.5, 0.2, 0.1)]       # directions that differ step to step
    seen = []

    def model(x, *a, **k):
        seen.append(x)
        return answers[len(seen) - 1]

    ts = torch.linspace(1.0, 0.0, 5)
    for i in range(4):
        out = wrap(model, x0, ts[i:i + 1], None, None, None, _to(i), latent_shapes=shapes)
        assert torch.equal(out, answers[i])                  # the answer itself is never edited
    assert torch.equal(seen[1], x0)                          # sigma 0.75 is the first step below 0.8: nothing was filed before it
    push = unpacked(seen[2] - x0, shapes)                    # step 1 filed an edit; step 2 got it
    assert not torch.allclose(push[0], torch.zeros_like(push[0]))
    assert torch.equal(push[1], torch.zeros_like(push[1]))   # sound untouched


def test_off_installs_nothing(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, _ = FunPackLoadModifiers.execute(tiny_h3.patcher, {"momentum": {"enabled": False}}).result
    assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)
