"""_compiler_disabled_for_forward_patches: Comfy's model compiler (aimdo malloc-graph)
crashes with "aimdo memory compile error" when a block's forward allocates a different
pattern than the compiler recorded. H3 representation steering while actually injecting
(not passive-capture) and block repeat both do that, so the sample_custom call that
actually runs the model must happen with comfy.cli_args.args.disable_comfy_compiler
forced True for exactly those cases -- and restored (or removed, if it never existed)
immediately after, so every other generation keeps the compiler's benefit."""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, ".")
import _comfy_stubs  # noqa: E402, F401
import comfy.cli_args  # noqa: E402
from samplers import FunPackLTXAVSceneChainSampler as S  # noqa: E402


class _FakeModel:
    """Enough of a real ModelPatcher for the install functions that need .clone() and
    .model_options (h3_repr_steering, block repeat) to actually succeed instead of
    hitting their `except Exception` fallback and silently handing the model back
    untouched -- which would make a "does it disable the compiler" test pass for the
    wrong reason (nothing installed, so of course nothing conflicts)."""
    def __init__(self):
        self.model_options = {}

    def clone(self):
        m = _FakeModel()
        m.model_options = dict(self.model_options)
        return m


def _run_chunk_with(monkeypatch, seen, refinement_key="fake_key", model=None, **kw):
    """Stubs sample_custom to record whether disable_comfy_compiler was set DURING the
    call, then _sample_chunk runs for real so the gating logic (h3_repr_steering /
    h3_repr_steering_strength / h3_block_repeat) executes exactly as it would in
    production."""
    def _fake_sample_custom(model, noise, cfg, smp, sigmas, pos, neg, samples, **k):
        seen.append(getattr(comfy.cli_args.args, "disable_comfy_compiler", "MISSING"))
        return samples

    monkeypatch.setattr(
        sys.modules["comfy.sample"], "prepare_noise",
        lambda samples, seed, **_: torch.zeros_like(samples))
    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom", _fake_sample_custom)
    latent = {"samples": torch.zeros(1, 2, 2, 2, 2)}
    sampler = type("F", (), {"extra_options": {}, "sampler_function": None})()
    S()._sample_chunk(model if model is not None else object(), sampler,
                      torch.tensor([1.0, 0.0]), 0, 1.0, [], [], latent,
                      refinement_key=refinement_key, **kw)


def test_reins_actively_injecting_disables_the_compiler(monkeypatch):
    seen = []
    _run_chunk_with(monkeypatch, seen, model=_FakeModel(),
                    h3_repr_steering=True, h3_repr_steering_strength=0.05)
    assert seen == [True]
    assert not hasattr(comfy.cli_args.args, "disable_comfy_compiler")  # cleaned up after


def test_reins_passive_capture_only_leaves_the_compiler_alone(monkeypatch):
    """model=_FakeModel() so _install_h3_repr_steering actually succeeds (clones, installs
    capture-only hooks) -- this must prove the strength/flag gate correctly keeps the
    compiler enabled on a REAL install, not merely that a failed install (object(), no
    .clone()) has nothing to disable it for either."""
    seen = []
    _run_chunk_with(monkeypatch, seen, model=_FakeModel(), h3_repr_steering=False,
                    h3_repr_steering_passive_capture=True)
    assert seen == ["MISSING"]


def test_reins_off_entirely_leaves_the_compiler_alone(monkeypatch):
    seen = []
    _run_chunk_with(monkeypatch, seen)
    assert seen == ["MISSING"]


def test_reins_enabled_but_zero_strength_leaves_the_compiler_alone(monkeypatch):
    """h3_repr_steering=True with strength 0 never enters REINS's own injection branch
    either (see _install_h3_repr_steering's `_strength > 0.0` gate) -- the compiler guard
    must agree, not just parrot the h3_repr_steering flag. model=_FakeModel() so the
    install actually succeeds (same reasoning as the passive-capture test above)."""
    seen = []
    _run_chunk_with(monkeypatch, seen, model=_FakeModel(),
                    h3_repr_steering=True, h3_repr_steering_strength=0.0)
    assert seen == ["MISSING"]


def test_block_repeat_disables_the_compiler(monkeypatch):
    seen = []
    _run_chunk_with(monkeypatch, seen, model=_FakeModel(),
                    h3_block_repeat="0-2", h3_block_repeat_times=1)
    assert seen == [True]
    assert not hasattr(comfy.cli_args.args, "disable_comfy_compiler")


def test_span_loop_that_refuses_to_install_leaves_the_compiler_alone(monkeypatch, capsys):
    """A non-contiguous selection makes _install_span_loop refuse and hand back the model
    UNTOUCHED ("blocks run once as usual") -- the block's forward is never repatched, so
    there is nothing for the compiler to conflict with. A gate built from the spec alone
    (non-empty h3_block_repeat) would disable the compiler here for no reason -- a real
    perf/VRAM regression on every call that happens to hit a refused install."""
    seen = []
    _run_chunk_with(monkeypatch, seen, h3_block_repeat="4,5,7", h3_block_repeat_times=1,
                    h3_block_repeat_span_loop=True)
    assert "blocks run once as usual" in capsys.readouterr().out
    assert seen == ["MISSING"]


def test_span_loop_with_no_block_list_leaves_the_compiler_alone(monkeypatch):
    """A contiguous selection still refuses if the model exposes no
    model.diffusion_model.blocks to loop over -- a second, separate refusal path from the
    non-contiguous one above, with the same "hand the model back untouched" shape. (Not
    asserting on the printed message here: _log.failed dedups by (tag, what), and the
    non-contiguous test above already said its piece for the same "H3 span loop" key this
    run -- what matters is the compiler gate, not the log line.)"""
    model = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace()))
    seen = []
    _run_chunk_with(monkeypatch, seen, model=model, h3_block_repeat="0-2",
                    h3_block_repeat_times=1, h3_block_repeat_span_loop=True)
    assert seen == ["MISSING"]


def test_restores_a_prior_value_instead_of_deleting_it(monkeypatch):
    """On a real build new enough to have the flag at all, disable_comfy_compiler already
    exists (default False) -- the guard must put back the PRIOR value, not just delete the
    attribute (which would be wrong on a real ComfyUI build)."""
    monkeypatch.setattr(comfy.cli_args.args, "disable_comfy_compiler", False, raising=False)
    seen = []
    _run_chunk_with(monkeypatch, seen, model=_FakeModel(),
                    h3_repr_steering=True, h3_repr_steering_strength=0.05)
    assert seen == [True]
    assert comfy.cli_args.args.disable_comfy_compiler is False


def test_reins_that_fails_to_install_leaves_the_compiler_alone(monkeypatch):
    """Same failure shape as the span-loop case above, for the OTHER installer: if
    _install_h3_repr_steering hits its own `except Exception` fallback (here, because
    the fake model has no .clone()) it hands back the model untouched -- no hooks, no
    patched block forward, nothing for the compiler to conflict with."""
    seen = []
    _run_chunk_with(monkeypatch, seen, model=object(),
                    h3_repr_steering=True, h3_repr_steering_strength=0.05)
    assert seen == ["MISSING"]


def test_crash_survives_the_sample_call_and_still_restores(monkeypatch):
    """A real generation can raise mid-sample (the exact aimdo crash this exists for) --
    the compiler flag must still be cleaned up, or every LATER generation this session
    silently loses the compiler even without REINS active."""
    def _boom(*a, **k):
        raise RuntimeError("aimdo memory compile error")

    monkeypatch.setattr(sys.modules["comfy.sample"], "prepare_noise",
                        lambda samples, seed, **_: torch.zeros_like(samples))
    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom", _boom)
    latent = {"samples": torch.zeros(1, 2, 2, 2, 2)}
    sampler = type("F", (), {"extra_options": {}, "sampler_function": None})()
    try:
        S()._sample_chunk(_FakeModel(), sampler, torch.tensor([1.0, 0.0]), 0, 1.0, [], [],
                          latent, refinement_key="fake_key", h3_repr_steering=True,
                          h3_repr_steering_strength=0.05)
    except RuntimeError:
        pass
    assert not hasattr(comfy.cli_args.args, "disable_comfy_compiler")
