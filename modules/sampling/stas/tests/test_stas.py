"""STAS: which rows and channels, in place, early only, on the real tiny H3's block."""

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


def test_the_target_rows_are_frame_zero_plus_every_frames_head_and_tail():
    from modules.sampling.stas import target_rows
    rows = target_rows(frames=3, per_frame=10, edge=0.1).tolist()
    assert rows == list(range(10)) + [10, 19, 20, 29]


def test_only_channels_far_above_the_mean_are_found_and_capped():
    from modules.sampling.stas import MAX_DIMS, ma_dims
    v = torch.randn(200, 16) * 0.1
    v[3, 5], v[9, 11] = 40.0, -30.0
    dims, peaks, ratios = ma_dims(v)
    assert sorted(dims.tolist()) == [5, 11] and (ratios > 50).all()
    wide = torch.randn(200, 64) * 0.01
    wide[0] = 100.0
    assert len(ma_dims(wide)[0]) == MAX_DIMS


def test_steering_sets_the_chosen_cells_to_alpha_times_the_peak_keeping_their_sign():
    from modules.sampling.stas import steer
    v = torch.zeros(6, 3)
    v[:, 1] = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    out = steer(v, torch.tensor([1, 3]), torch.tensor([1]), torch.tensor([6.0]), 2.0)
    assert out is v and v[1, 1] == -12.0 and v[3, 1] == -12.0 and v[0, 1] == 1.0 and v[:, 0].abs().sum() == 0


def _load(tiny, **values):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    settings = {"taste": {"key": "fox"}, "stas": {"enabled": True, "block": 1, **values}}
    patched, status = FunPackLoadModifiers.execute(tiny.patcher, settings).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    outer = [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws]
    return patched, wrap, outer, status


def _forward(tiny, patched, wrap, outer, i, sched):
    from conftest import packed_av
    x0, shapes = packed_av(tiny.video, tiny.audio)

    def executor(x, t, *a, **k):
        tiny.run(patched, sigma=float(sched[i]), sigmas=sched)
        return x

    def call():
        return wrap(executor, x0, sched[i:i + 1], None, None, None, {}, latent_shapes=shapes)

    run = call
    for o in reversed(outer):
        run = (lambda w, inner: (lambda: w(lambda: inner())))(o, run)
    return run()


def _block_output(tiny, patched, block):
    """What block `block` hands on, read by a hook installed outside STAS's."""
    from core import dit_hooks
    seen = []
    dit_hooks.add_block_hook(patched, "funpack.spy", block,
                             lambda args, extra: (seen.append(extra["original_block"](args)["img"].clone()),
                                                  {"img": seen[-1]})[1])
    return seen


def test_with_a_massive_channel_the_first_steps_are_steered_and_later_ones_are_not(tiny_h3, monkeypatch):
    from modules.sampling import stas
    fake = (torch.tensor([2]), torch.tensor([5.0]), torch.tensor([99.0]))
    monkeypatch.setattr(stas, "ma_dims", lambda video: fake)
    patched, wrap, outer, _ = _load(tiny_h3, mode="manual", alpha=2.0)
    seen = _block_output(tiny_h3, patched, 1)
    sched = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    for i in (0, 4):
        _forward(tiny_h3, patched, wrap, outer, i, sched)
    early, late = seen
    steered = lambda h: torch.isclose(h.abs().float(), torch.tensor(10.0), atol=0.05).any()
    assert steered(early) and not steered(late)         # alpha*peak = 10, first steps only


def test_a_model_without_massive_channels_says_so(tiny_h3):
    from core import log
    patched, wrap, outer, _ = _load(tiny_h3, mode="manual")
    log.new_run()
    _forward(tiny_h3, patched, wrap, outer, 0, torch.tensor([1.0, 0.5, 0.0]))
    assert any("Inactive" in e["message"] and "nothing steered" in e["message"] for e in log.history())


def test_learned_mode_banks_alpha_when_it_steered(tiny_h3, monkeypatch):
    from modules.sampling import stas
    from modules.system.taste import store
    monkeypatch.setattr(stas, "ma_dims", lambda video: (torch.tensor([2]), torch.tensor([5.0]), torch.tensor([99.0])))
    patched, wrap, outer, _ = _load(tiny_h3)
    _forward(tiny_h3, patched, wrap, outer, 0, torch.tensor([1.0, 0.8, 0.6, 0.4, 0.0]))
    assert store.rate("run-1", "liked")["recorded"] == ["stas"]
    row = store.load("fox", "stas")["rows"][0]["rows"]
    assert 0.5 <= float(row["v"]) <= 3.0 and int(row["b"]) == 1


def test_a_block_outside_the_model_is_refused_and_said(tiny_h3):
    from core import log
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    log.new_run()
    FunPackLoadModifiers.execute(tiny_h3.patcher, {"taste": {"key": "fox"}, "stas": {"enabled": True, "block": 40}})
    assert any("not a block of this model" in e["message"] for e in log.history())
