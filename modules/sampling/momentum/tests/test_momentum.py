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


def test_a_carried_edit_moves_the_next_input_exactly_like_v4s_blended_direction(tiny_h3):
    from conftest import packed_av, unpacked
    k, decay, sigmas = 0.7, 0.5, torch.linspace(1.0, 0.0, 5)
    _p, wrap, _ = _load(tiny_h3, strength=k, decay=decay, below_sigma=0.8)
    torch.manual_seed(0)
    x0, shapes = packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))
    answers = [x0 * f for f in (0.9, 0.5, 0.2, 0.1)]
    seen = []

    def model(x, *a, **kw):
        seen.append(x)
        return answers[len(seen) - 1]

    for i in range(3):
        wrap(model, x0, sigmas[i:i + 1], None, None, None, _to(i), latent_shapes=shapes)
    v = lambda t: unpacked(t, shapes)[0]
    d0, d1 = (v(x0) - v(answers[0])) / sigmas[0], (v(x0) - v(answers[1])) / sigmas[1]
    ema1 = decay * d0 + (1 - decay) * d1                                  # the memory after step 1
    v4_delta = (sigmas[2] - sigmas[1]) * k * (ema1 - d1)                  # x_next moves by dt * (blended - d)
    assert torch.allclose(v(seen[2] - x0), v4_delta, atol=1e-5)


def test_a_plain_latent_edit_on_the_answer_equals_v4s_step_and_skips_the_last_step(tiny_h3):
    k, decay, sigmas = 0.6, 0.5, torch.linspace(1.0, 0.0, 5)
    _p, wrap, _ = _load(tiny_h3, strength=k, decay=decay, below_sigma=0.9)
    x = torch.randn(1, 4, 3, 8, 8)
    outs = [x * f for f in (0.9, 0.5, 0.2, 0.1)]
    got = [wrap(lambda *a, o=outs[i], **kw: o, x, sigmas[i:i + 1], None, None, None, _to(i)) for i in range(4)]
    d = [(x - outs[i]) / sigmas[i] for i in range(4)]
    ema = d[0]
    assert torch.allclose(got[0], outs[0])                   # above the threshold: nothing
    for i in (1, 2):
        ema = decay * ema + (1 - decay) * d[i]
        blended = d[i] + k * (ema - d[i])
        assert torch.allclose(got[i], x - sigmas[i] * blended, atol=1e-5)    # v4: x_next = x + dt * blended
    assert torch.equal(got[3], outs[3])                      # v4 never edits the step that lands on 0


def test_clips_cut_into_context_windows_are_refused_in_words(tiny_h3):
    from conftest import packed_av
    from core import log
    log.new_run()
    wrap = _load(tiny_h3, strength=1.0)[1]
    x0, shapes = packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))
    to = {**_to(1), "context_window": object()}
    out = wrap(lambda x, *a, **k: x0 * 0.5, x0, torch.tensor([0.75]), None, None, None, to, latent_shapes=shapes)
    assert torch.equal(out, x0 * 0.5)
    assert any("context windows" in e["message"] for e in log.history())


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


def test_a_second_pass_that_restarts_lower_still_starts_with_an_empty_memory(tiny_h3):
    from conftest import packed_av
    wrap = _load(tiny_h3, strength=1.0, decay=0.5, below_sigma=1.0)[1]
    x0, shapes = packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))
    first = [0.9, 0.5, 0.2]
    seen = []

    def run(steps, f):
        ts = torch.linspace(1.0, 0.0, steps + 1)
        for i in range(steps):
            wrap(lambda x, *a, **k: seen.append(x) or x0 * f[i], x0, ts[i:i + 1], None, None, None, _to(i, steps),
                 latent_shapes=shapes)
    run(4, first + [0.1])
    seen.clear()
    run(4, first + [0.1])
    twice = [s for s in seen]
    seen.clear()
    fresh = _load(tiny_h3, strength=1.0, decay=0.5, below_sigma=1.0)[1]
    wrap = fresh
    run(4, first + [0.1])
    assert all(torch.allclose(a, b) for a, b in zip(twice, seen))


def test_a_second_order_sampler_on_a_plain_latent_is_said_and_left_alone(tiny_h3):
    from core import log
    log.new_run()
    _p, wrap, _ = _load(tiny_h3, strength=1.0, decay=0.5, below_sigma=1.0)
    x = torch.randn(1, 4, 3, 8, 8)
    sig = torch.linspace(1.0, 0.0, 5)
    outs = []
    for i in (0, 1, 1, 2, 2, 3):                              # heun: each step's corrector lands on the next sigma
        o = x * (0.9 - 0.1 * len(outs))
        outs.append(wrap(lambda *a, o=o, **kw: o, x, sig[i:i + 1], None, None, None, _to(i)))
    assert any("more than once per step" in e["message"] for e in log.history())
    assert torch.equal(outs[-1], x * (0.9 - 0.5))             # nothing edited once it is known
