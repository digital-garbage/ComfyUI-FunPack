"""Late-branch guidance on the real tiny H3: which blocks the weak copy runs, and where the push goes."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-1")


def _load(tiny, **values):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    settings = {"taste": {"key": "fox"}, "late_branch": {"enabled": True, "block": 2, **values}}
    patched, status = FunPackLoadModifiers.execute(tiny.patcher, settings).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    outer = [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws]
    return patched, wrap, outer, status


def _executor(tiny, patched, shapes):
    """The real tiny H3 forward, packed the way comfy hands it back."""
    from conftest import packed_av
    calls = []

    def executor(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kw):
        calls.append(dict(transformer_options or {}))
        sigma = float(t.max())
        video, audio = tiny.run(patched, sigma=sigma, sigmas=torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
                                options={k: v for k, v in (transformer_options or {}).items()
                                         if k in ("funpack_weak_branch", "funpack_probe")})
        return packed_av(video, audio)[0]
    return executor, calls


def _run(outer, fn):
    call = fn
    for o in reversed(outer):
        call = (lambda w, inner: (lambda: w(lambda: inner())))(o, call)
    return call()


def _to(i, steps=4, **extra):
    s = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    return {"sample_sigmas": s, "sigmas": s[i:i + 1], "cond_or_uncond": [0], **extra}


def _setup(tiny_h3, **values):
    from conftest import packed_av
    patched, wrap, outer, status = _load(tiny_h3, **values)
    x0, shapes = packed_av(tiny_h3.video, tiny_h3.audio)
    executor, calls = _executor(tiny_h3, patched, shapes)
    return patched, wrap, outer, x0, shapes, executor, calls


def test_the_weak_copy_runs_only_the_blocks_after_the_branch_and_the_answer_stays_the_models(tiny_h3):
    patched, wrap, outer, x0, shapes, executor, _calls = _setup(tiny_h3, mode="manual", strength=1.0)
    ran = []
    blocks = patched.model.diffusion_model.blocks
    for i, b in enumerate(blocks):
        orig = b.forward
        b.forward = (lambda orig, i: (lambda *a, **k: (ran.append(i), orig(*a, **k))[1]))(orig, i)

    def go(i):
        t = torch.tensor([[1.0, 0.75, 0.5, 0.25][i]])
        return wrap(executor, x0, t, None, None, None, _to(i), latent_shapes=shapes)

    answers = _run(outer, lambda: [go(i) for i in range(4)])
    n = len(blocks)
    plain = [i for i in ran]
    # steps 0-2 each: all blocks once (normal) + only blocks after the branch (weak); step 3 is final: normal only
    assert plain.count(n - 1) == 3 * 2 + 1 and plain.count(0) == 4
    assert plain.count(2) == 4                          # the branch block runs only in the normal pass
    assert all(a.shape == x0.shape for a in answers)


def test_the_push_rides_into_the_next_input_in_the_picture_rows_only(tiny_h3):
    patched, wrap, outer, x0, shapes, executor, _ = _setup(tiny_h3, mode="manual", strength=1.0)
    seen = []

    def spy(x, t, *a, **k):
        seen.append(x.clone())
        return executor(x, t, *a, **k)

    def go(i):
        t = torch.tensor([[1.0, 0.75, 0.5, 0.25][i]])
        return wrap(spy, x0, t, None, None, None, _to(i), latent_shapes=shapes)

    _run(outer, lambda: [go(i) for i in range(4)])
    # normal inputs of steps 0..3 are the 1st, 3rd, 5th, 7th spy entries (weak passes in between)
    inputs = [seen[0], seen[2], seen[4], seen[6]]
    assert torch.equal(inputs[0], x0)                   # step 1: nothing pushed yet
    from conftest import unpacked
    for i in (1, 2, 3):                                 # every step but the last makes a push
        push = unpacked(inputs[i] - x0, shapes)
        assert not torch.allclose(push[0], torch.zeros_like(push[0]))
        assert torch.equal(push[1], torch.zeros_like(push[1]))   # sound untouched


def test_probe_calls_and_negative_calls_are_left_unguided_and_weak_calls_are_marked(tiny_h3):
    from core import dit_hooks
    patched, wrap, outer, x0, shapes, executor, calls = _setup(tiny_h3, mode="manual", strength=1.0)

    def go(to):
        return wrap(executor, x0, torch.tensor([0.5]), None, None, None, to, latent_shapes=shapes)

    _run(outer, lambda: go(_to(1, **{dit_hooks.PROBE: True})))
    assert len(calls) == 1
    calls.clear()
    _run(outer, lambda: go(_to(1, cond_or_uncond=[0, 1])))
    assert len(calls) == 1
    calls.clear()
    _run(outer, lambda: go(_to(1)))
    assert len(calls) == 2 and not calls[0].get("funpack_weak_branch")
    assert calls[1]["funpack_weak_branch"] and dit_hooks.probing(calls[1])


def test_learned_mode_banks_strength_effect_and_branch_after_a_felt_push_only(tiny_h3, monkeypatch):
    from core import input_steer
    from modules.system.taste import store
    monkeypatch.setattr(input_steer, "FLOOR", 0.0)
    patched, wrap, outer, x0, shapes, executor, _ = _setup(tiny_h3)

    def go(i):
        t = torch.tensor([[1.0, 0.75, 0.5, 0.25][i]])
        return wrap(executor, x0, t, None, None, None, _to(i), latent_shapes=shapes)

    _run(outer, lambda: [go(i) for i in range(4)])
    assert store.rate("run-1", "liked")["recorded"] == ["late_branch"]
    row = store.load("fox", "late_branch")["rows"][0]["rows"]
    assert 0.0 <= float(row["v"]) <= 1.5 and float(row["e"]) > 0 and int(row["b"]) == 2


def test_a_push_too_small_to_feel_teaches_nothing(tiny_h3, monkeypatch):
    from core import input_steer
    from modules.system.taste import store
    monkeypatch.setattr(input_steer, "FLOOR", 10.0)          # nothing can be big enough
    patched, wrap, outer, x0, shapes, executor, _ = _setup(tiny_h3)
    _run(outer, lambda: [wrap(executor, x0, torch.tensor([[1.0, 0.75, 0.5, 0.25][i]]), None, None, None,
                              _to(i), latent_shapes=shapes) for i in range(4)])
    assert store.rate("run-1", "liked")["why"]                # nothing was captured


def test_strength_zero_runs_no_weak_copy_and_the_mix_pushes_away_from_the_weak_one():
    from modules.sampling.late_branch import mix
    n, w = torch.tensor([2.0]), torch.tensor([1.0])
    assert float(mix(n, w, 0.5)) == 2.5 and float(mix(n, w, 0.0)) == 2.0


def test_manual_strength_zero_makes_one_call_per_step(tiny_h3):
    patched, wrap, outer, x0, shapes, executor, calls = _setup(tiny_h3, mode="manual", strength=0.0)
    _run(outer, lambda: wrap(executor, x0, torch.tensor([0.5]), None, None, None, _to(1), latent_shapes=shapes))
    assert len(calls) == 1


def test_a_branch_block_outside_the_model_is_refused_and_said(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from core import log
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    log.new_run()
    patched, _ = FunPackLoadModifiers.execute(tiny_h3.patcher, {
        "taste": {"key": "fox"}, "late_branch": {"enabled": True, "block": 43}}).result
    assert any("not a branch point" in e["message"] for e in log.history())
    assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)


def test_a_failure_after_the_weak_copy_returns_the_models_own_answer_not_the_weak_one(tiny_h3, monkeypatch):
    from modules.sampling import late_branch
    patched, wrap, outer, x0, shapes, executor, calls = _setup(tiny_h3, mode="manual", strength=1.0)
    plain = executor(x0, torch.tensor([0.5]))
    monkeypatch.setattr(late_branch, "mix", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    out = _run(outer, lambda: wrap(executor, x0, torch.tensor([0.5]), None, None, None, _to(1), latent_shapes=shapes))
    assert torch.equal(out, plain)
