"""Camera move: the warp maths and the wrapper on packed H3 latents."""

import types

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def test_a_pan_shifts_the_picture_and_uncovered_cells_get_noise_at_the_step_level():
    from modules.sampling.camera_move import Move, warp
    video = torch.arange(1 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    out = warp(video, Move(pan_x=0.5), sigma=0.0, generator=torch.Generator().manual_seed(0))
    assert torch.equal(out[:, 0], video[:, 0])                         # frame 0 has not moved yet
    assert torch.equal(out[:, 2][..., :2], video[:, 2][..., 2:])        # last frame: shifted 2 cells left
    noisy = warp(video, Move(pan_x=0.5), sigma=0.5, generator=torch.Generator().manual_seed(0))
    assert not torch.equal(noisy[:, 2][..., 2:], torch.zeros(1, 4, 2))  # uncovered cells are filled


def test_unwarp_puts_shown_cells_back_and_keeps_the_original_where_nothing_was_shown():
    from modules.sampling.camera_move import Move, unwarp, warp
    move = Move(pan_x=0.5)
    video = torch.randn(2, 3, 4, 4)
    moved = warp(video, move, 0.0, torch.Generator().manual_seed(0))
    back = unwarp(moved, video, move)
    assert torch.equal(back, video)                                     # a model that echoes its input


def test_still_moves_and_nan_values_are_refused_or_clamped():
    from modules.sampling.camera_move import Move, finite
    assert Move().still() and not Move(zoom=1.2).still()
    assert finite(float("nan"), 0.0, -1, 1) == 0.0 and finite(5, 0.0, -1, 1) == 1.0
    assert Move(step=99).first_step(4) == 3 and Move(step=1).first_step(4) == 0


SHAPES = [torch.Size([1, 4, 4, 8, 8]), torch.Size([1, 3, 5])]


class Guider:
    def __init__(self):
        self.inner_model = types.SimpleNamespace(latent_shapes=SHAPES)


def _load(tiny, **values):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, status = FunPackLoadModifiers.execute(tiny.patcher, {
        "camera_move": {"enabled": True, "pan_x": 0.5, "step": 2, **values}}).result
    sample = [w for ws in patched.wrappers[WrappersMP.SAMPLER_SAMPLE].values() for w in ws]
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    return sample[-1] if sample else None, wrap, status


def _packed():
    from conftest import packed_av
    torch.manual_seed(5)
    return packed_av(torch.randn(1, 4, 4, 8, 8), torch.randn(1, 3, 5))


def _to(i, steps=4):
    s = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": s, "sigmas": s[i:i + 1]}


def _steps(sample, wrap, x0, shapes, latent=None, to=_to, steps=4, executor=None):
    """Drive the wrapper the way a sampler does: the sampler wrapper first, then every step."""
    class Ex:
        def __call__(self, *a, **k):
            return a[4]

    seen = []

    def echo(x, t, *a, **k):
        seen.append(x.clone())
        return x

    sample(Ex(), Guider(), None, {"seed": 7}, None, x0, torch.zeros_like(x0) if latent is None else latent, None, True)
    sched = torch.linspace(1.0, 0.0, steps + 1)
    outs = [wrap(executor or echo, x0, sched[i:i + 1], None, None, None, to(i, steps), latent_shapes=shapes)
            for i in range(steps)]
    return outs, seen


def test_a_model_that_echoes_its_input_sees_the_moved_picture_but_returns_the_original_until_the_last_step(tiny_h3):
    sample, wrap, status = _load(tiny_h3)
    assert "pan right 0.50" in status
    x0, shapes = _packed()
    outs, seen = _steps(sample, wrap, x0, shapes)
    from conftest import unpacked
    assert torch.equal(seen[0], x0) and torch.equal(outs[0], x0)        # before 'from step'
    assert not torch.equal(seen[1], x0) and torch.equal(outs[1], x0)    # moved in, moved back out
    assert torch.equal(unpacked(seen[1], shapes)[1], unpacked(x0, shapes)[1])    # sound untouched
    assert not torch.equal(outs[3], x0)                                 # the last call stays moved


def test_standing_down_cases_are_declared_and_leave_the_call_alone(tiny_h3):
    from core import log
    sample, wrap, _ = _load(tiny_h3)
    x0, shapes = _packed()
    log.new_run()
    outs, _ = _steps(sample, wrap, x0, shapes, latent=torch.ones_like(x0))
    assert all(torch.equal(o, x0) for o in outs)
    assert any("not empty" in e["message"] for e in log.history())
    log.new_run()
    outs, _ = _steps(sample, wrap, x0, shapes, to=lambda i, n: {**_to(i, n), "context_window": object()})
    assert all(torch.equal(o, x0) for o in outs) and any("context windows" in e["message"] for e in log.history())


def test_a_sampler_that_repeats_a_step_is_declared_and_the_rest_of_the_run_is_left_alone(tiny_h3):
    from core import log
    sample, wrap, _ = _load(tiny_h3, step=1)
    x0, shapes = _packed()
    log.new_run()
    class Ex:
        def __call__(self, *a, **k):
            return a[4]
    sample(Ex(), Guider(), None, {"seed": 1}, None, x0, torch.zeros_like(x0), None, True)
    sched = torch.linspace(1.0, 0.0, 5)
    calls = []
    for i in (0, 1, 1, 2):
        wrap(lambda x, t, *a, **k: (calls.append(x), x)[1], x0, sched[i:i + 1], None, None, None, _to(i), latent_shapes=shapes)
    assert any("more than once per step" in e["message"] for e in log.history())
    assert torch.equal(calls[-1], x0)


def test_no_move_is_set_and_the_wrapper_is_not_installed_and_says_so(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from core import log
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    log.new_run()
    patched, _ = FunPackLoadModifiers.execute(tiny_h3.patcher, {"camera_move": {"enabled": True}}).result
    assert any("nothing to move" in e["message"] for e in log.history())
    assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)


def test_the_first_moved_call_is_marked_so_an_old_frame_push_is_not_added(tiny_h3):
    sample, wrap, _ = _load(tiny_h3)
    x0, shapes = _packed()
    flags = []

    def spy(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **k):
        flags.append(bool((transformer_options or {}).get("funpack_frame_change")))
        return x

    _steps(sample, wrap, x0, shapes, executor=spy)
    assert flags == [False, True, False, False]            # 'from step 2' of 4


def test_a_probe_call_is_neither_moved_nor_counted_as_a_step(tiny_h3):
    sample, wrap, _ = _load(tiny_h3, step=1)
    x0, shapes = _packed()
    seen = []
    outs, _ = _steps(sample, wrap, x0, shapes, to=lambda i, n: {**_to(i, n), "funpack_probe": True},
                     executor=lambda x, *a, **k: (seen.append(x), x)[1])
    assert all(torch.equal(s, x0) for s in seen)


def test_a_repeated_step_never_logs_both_active_and_inactive(tiny_h3):
    from core import log
    sample, wrap, _ = _load(tiny_h3, step=1)
    x0, shapes = _packed()
    sched = torch.linspace(1.0, 0.0, 5)

    class Ex:                                           # the sampler: runs the steps inside the call
        def __call__(self, *a, **k):
            for i in (0, 1, 1, 2):
                wrap(lambda x, *a, **k: x, x0, sched[i:i + 1], None, None, None, _to(i), latent_shapes=shapes)
            return a[4]

    log.new_run()
    sample(Ex(), Guider(), None, {"seed": 1}, None, x0, torch.zeros_like(x0), None, True)
    msgs = [e["message"] for e in log.history() if e["source"] == "FunPack Camera move"]
    assert any("more than once" in m for m in msgs) and not any(m.startswith("Active") for m in msgs)
