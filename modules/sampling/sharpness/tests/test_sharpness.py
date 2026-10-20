import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _load(tiny, **v):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, status = FunPackLoadModifiers.execute(
        tiny.patcher, {"sharpness": {"enabled": True, **v}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    return patched, wrap, status


def _to(i, steps=4):
    s = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": s, "sigmas": s[i:i + 1]}


def test_off_installs_nothing(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, _ = FunPackLoadModifiers.execute(tiny_h3.patcher, {"sharpness": {"enabled": False}}).result
    assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)


def test_a_plain_latent_is_pushed_away_from_the_previous_prediction_in_the_last_part_only(tiny_h3):
    _p, wrap, status = _load(tiny_h3, amount=0.5, last_part=0.5)
    assert "sharpness" in status
    x = torch.randn(1, 4, 3, 8, 8)
    outs = [x * f for f in (0.9, 0.5, 0.2, 0.1)]
    ts = torch.linspace(1.0, 0.0, 5)
    got = [wrap(lambda *a, o=outs[i], **kw: o, x, ts[i:i + 1], None, None, None, _to(i)) for i in range(4)]
    assert torch.equal(got[0], outs[0]) and torch.equal(got[1], outs[1])      # outside the last half
    for i in (2, 3):
        assert torch.allclose(got[i], outs[i] + 0.25 * (outs[i] - outs[i - 1]), atol=1e-5)


def test_a_packed_model_carries_the_edit_into_the_next_input_and_never_edits_the_last_step(tiny_h3):
    from conftest import packed_av, unpacked
    _p, wrap, _ = _load(tiny_h3, amount=1.0, last_part=1.0)
    x0, shapes = packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))
    answers = [x0 * f for f in (0.9, 0.5, 0.2, 0.1)]
    seen = []

    def model(x, *a, **k):
        seen.append(x)
        return answers[len(seen) - 1]

    ts = torch.linspace(1.0, 0.0, 5)
    for i in range(4):
        out = wrap(model, x0, ts[i:i + 1], None, None, None, _to(i), latent_shapes=shapes)
        assert torch.equal(out, answers[i])
    assert torch.equal(seen[1], x0)                        # nothing to compare on step 0
    push = unpacked(seen[2] - x0, shapes)
    assert not torch.allclose(push[0], torch.zeros_like(push[0]))
    assert torch.equal(push[1], torch.zeros_like(push[1]))  # sound untouched
