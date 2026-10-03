import types

import pytest
import torch

from core import patching

KEY = "funpack.temporal_style"


class Cond:
    def __init__(self, cond):
        self.cond = cond


def _install(tiny, style, old=None):
    from modules.sampling.temporal_style import install
    p = tiny.patcher.clone()
    if old is not None:
        p.set_model_unet_function_wrapper(old)
    g = patching.GuardedPatcher(p, KEY, patching.Dropped())
    return p, install(g, {"style": style}, key=KEY)


def _call(patched, **extra):
    seen = {}

    def apply_fn(x, t, **c):
        seen["frame_rate"] = c["frame_rate"].cond
        seen["x"] = x
        return x

    args = {"input": torch.zeros(1, 1, 4), "timestep": torch.tensor([0.5]),
            "c": {"frame_rate": Cond(25.0), **extra}}
    patched.model_options["model_function_wrapper"](apply_fn, args)
    return seen


def test_natural_installs_nothing(tiny_ltx):
    p, note = _install(tiny_ltx, "natural")
    assert note is None and "model_function_wrapper" not in p.model_options


@pytest.mark.parametrize("style, mult", [("accelerate", 1.35), ("decelerate", 0.72), ("freeze", 2.0)])
def test_the_frame_rate_the_model_sees_is_scaled(tiny_ltx, style, mult):
    p, note = _install(tiny_ltx, style)
    assert _call(p)["frame_rate"] == pytest.approx(25.0 * mult) and style in note


def test_pulse_and_rapid_follow_the_schedules_progress(tiny_ltx):
    sched = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    def at(style, step):
        p, _ = _install(tiny_ltx, style)
        return _call(p, transformer_options={"sample_sigmas": sched, "sigmas": sched[step:step + 1]})["frame_rate"] / 25.0

    assert at("rapid_start", 0) == pytest.approx(0.65) and at("rapid_start", 2) == pytest.approx(1.0)
    assert at("rapid_end", 5) == pytest.approx(0.65) and at("rapid_end", 1) == pytest.approx(1.0)
    assert at("pulse", 0) == pytest.approx(0.88)


def test_an_earlier_wrapper_still_runs_and_comes_back_when_ours_is_stripped(tiny_ltx):
    ran = []

    def earlier(apply_fn, args):
        ran.append(args["c"]["frame_rate"].cond)
        return apply_fn(args["input"], args["timestep"], **args["c"])

    p, _ = _install(tiny_ltx, "freeze", old=earlier)
    _call(p)
    assert ran == [50.0]                                  # ours scaled, then the earlier one ran
    patching.strip(p, KEY)
    assert p.model_options["model_function_wrapper"] is earlier


def test_a_failing_wrapper_is_dropped_and_the_model_still_runs(tiny_ltx):
    from modules.sampling.temporal_style import install
    p = tiny_ltx.patcher.clone()
    dropped = patching.Dropped()
    install(patching.GuardedPatcher(p, KEY, dropped), {"style": "freeze"}, key=KEY)

    class Broken:
        @property
        def cond(self):
            raise ValueError("no frame rate")

    ran = []
    args = {"input": torch.zeros(1, 1, 4), "timestep": torch.tensor([0.5]), "c": {"frame_rate": Broken()}}
    out = p.model_options["model_function_wrapper"](lambda x, t, **c: ran.append(1) or x, args)
    assert ran == [1] and out is args["input"] and "funpack.temporal_style" in dropped


def _loop_args(video, audio, sigma=0.5):
    packed = torch.cat([video.reshape(1, 1, -1), audio.reshape(1, 1, -1)], dim=-1)
    return {"input": packed, "timestep": torch.tensor([sigma]),
            "c": {"latent_shapes": [video.shape, audio.shape]}}


def test_the_loop_roll_is_undone_on_the_way_out_and_moves_the_seam_in_between(tiny_ltx):
    p, note = _install(tiny_ltx, "loop")
    video, audio = torch.randn(1, 8, 8, 2, 2), torch.randn(1, 4, 8, 3)
    seen = {}

    def apply_fn(x, t, **c):
        seen["x"] = x.clone()
        return x * 2.0

    args = _loop_args(video, audio)
    out = p.model_options["model_function_wrapper"](apply_fn, args)
    assert torch.allclose(out, args["input"] * 2.0)         # rolled in, rolled back out: canonical orientation
    assert not torch.equal(seen["x"], args["input"])         # the model saw it rolled
    assert "loop" in note


def test_the_loop_leaves_the_noisy_early_steps_and_short_clips_alone(tiny_ltx):
    p, _ = _install(tiny_ltx, "loop")
    seen = {}

    def apply_fn(x, t, **c):
        seen["x"] = x.clone()
        return x

    early = _loop_args(torch.randn(1, 8, 8, 2, 2), torch.randn(1, 4, 8, 3), sigma=0.99)
    p.model_options["model_function_wrapper"](apply_fn, early)
    assert torch.equal(seen["x"], early["input"])
    short = _loop_args(torch.randn(1, 8, 2, 2, 2), torch.randn(1, 4, 2, 3))
    p.model_options["model_function_wrapper"](apply_fn, short)
    assert torch.equal(seen["x"], short["input"])
