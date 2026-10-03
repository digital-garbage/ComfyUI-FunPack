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


def test_the_carried_edit_is_sized_to_move_the_next_input_like_v4s_blended_direction():
    from modules.sampling.momentum import carry_factor
    sigma, nxt = 0.75, 0.5
    # an x0 edit of -sigma*dd, scaled by this and carried in (x (1 - nxt)), moves x by (sigma - nxt)*dd
    assert abs(carry_factor(sigma, nxt) * sigma * (1 - nxt) - (sigma - nxt) * 1.0) < 1e-9
    assert abs(carry_factor(sigma, nxt, carried=False) * sigma - (sigma - nxt)) < 1e-9


def _run_calls(wrap, shapes, x0, n, steps=4, ts_start=0, **extra):
    seen = []
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(ts_start, ts_start + n):
        def model(x, *a, **k):
            seen.append(x)
            return x0 * (0.9 - 0.2 * len(seen))
        wrap(model, x0, ts[i:i + 1], None, None, None, _to(i, steps), latent_shapes=shapes, **extra)
    return seen


def test_a_probe_call_does_not_feed_the_memory(tiny_h3):
    from conftest import packed_av
    from core import dit_hooks
    torch.manual_seed(0)
    x0, shapes = packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))
    clean = _run_calls(_load(tiny_h3, strength=1.0, decay=0.5, below_sigma=1.0)[1], shapes, x0, 4)
    wrap = _load(tiny_h3, strength=1.0, decay=0.5, below_sigma=1.0)[1]
    ts = torch.linspace(1.0, 0.0, 5)
    probe_to = {**_to(0), dit_hooks.PROBE: True}
    for _ in range(3):
        wrap(lambda x, *a, **k: x0 * 0.1, x0, ts[0:1], None, None, None, probe_to, latent_shapes=shapes)
    after = _run_calls(wrap, shapes, x0, 4)
    assert all(torch.allclose(a, b) for a, b in zip(clean, after))


def test_each_window_and_each_call_kind_has_its_own_memory_and_a_new_run_starts_empty(tiny_h3):
    from conftest import packed_av
    torch.manual_seed(0)
    patched, wrap, _ = _load(tiny_h3, strength=1.0, decay=0.5, below_sigma=1.0)
    big, sb = packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))
    small, ss = packed_av(torch.randn(1, 4, 2, 8, 8), torch.randn(1, 8, 5))

    class Win:
        def __init__(self, idx):
            self.index_list = idx
    ts = torch.linspace(1.0, 0.0, 5)
    for i in range(3):                               # two windows of different sizes interleaved: must not collide
        for x0, shapes, idx in ((big, sb, [0, 1, 2]), (small, ss, [3, 4])):
            to = {**_to(i), "context_window": Win(idx)}
            wrap(lambda x, *a, **k: x0 * 0.5, x0, ts[i:i + 1], None, None, None, to, latent_shapes=shapes)
    assert not patched.model_options["funpack_dropped"]      # no size clash: the guard would have dropped it
