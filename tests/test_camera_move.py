"""Camera move (camera_move.py): pan/zoom of the running latent, with the sampler's own
state kept in the original frame and only the last prediction left moved."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import camera_move as cm  # noqa: E402


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_pan_right_moves_the_picture_left_by_the_stated_share():
    x = torch.arange(20.0).repeat(2, 9, 12, 1)              # value = column
    out = cm.warp(x, cm.Move(pan_x=0.5), 0.0, _gen())      # 10 cells over the clip
    assert torch.equal(out[:, 0], x[:, 0])                  # frame 0 untouched
    assert torch.equal(out[:, -1][:, :, :10], x[:, -1][:, :, 10:])
    assert torch.equal(out[:, 4][:, :, :15], x[:, 4][:, :, 5:])
    assert float(out[:, -1][:, :, 10:].abs().max()) == 0.0   # uncovered = fresh * sigma(0)


def test_pan_down_moves_the_picture_up():
    x = torch.arange(12.0).view(1, 1, 12, 1).repeat(2, 9, 1, 20)
    out = cm.warp(x, cm.Move(pan_y=0.5), 0.0, _gen())
    assert torch.equal(out[:, -1][:, :6, :], x[:, -1][:, 6:, :])


def test_zoom_in_and_out_and_the_focus_point():
    x = torch.randn(3, 5, 12, 20, generator=_gen(1))
    zin = cm.warp(x, cm.Move(zoom=2.0), 0.0, _gen())
    assert torch.equal(zin[:, -1][:, 0, 0], x[:, -1][:, 3, 5])      # (0.5-6)/2+6, (0.5-10)/2+10
    assert torch.equal(zin[:, -1][:, 6, 10], x[:, -1][:, 6, 10])   # the focus stays
    zout = cm.warp(x, cm.Move(zoom=0.5), 0.0, _gen())
    assert torch.equal(zout[:, -1][:, 6, 11], x[:, -1][:, 7, 13])
    left = cm.warp(x, cm.Move(zoom=2.0, focus_x=0.0), 0.0, _gen())
    assert torch.equal(left[:, -1][:, 6, 0], x[:, -1][:, 6, 0])     # aimed at the left edge


def test_shifting_white_noise_leaves_it_white():
    """The reason for moving the latent instead of the starting noise."""
    x = torch.randn(24, 9, 48, 84, generator=_gen(2))
    out = cm.warp(x, cm.Move(pan_x=0.3, pan_y=-0.2), 0.9, _gen(3))
    assert abs(float(out.std()) - 1.0) < 0.05
    a, b = out[..., :-1], out[..., 1:]
    assert abs(float((a * b).mean() / out.std() ** 2)) < 0.02        # neighbours unrelated
    # and frames stay independent of each other
    assert abs(float((out[:, 3] * out[:, 4]).mean())) < 0.02


def test_unwarp_puts_the_covered_cells_back_and_leaves_the_rest():
    x = torch.randn(2, 9, 12, 20, generator=_gen(4))
    move = cm.Move(pan_x=0.5)
    back = cm.unwarp(cm.warp(x, move, 0.0, _gen()), x, move)
    assert torch.equal(back[:, -1][:, :, 10:], x[:, -1][:, :, 10:])  # covered by the move: same picture
    assert torch.equal(back[:, -1][:, :, :10], x[:, -1][:, :, :10])


class _Blur:
    """A shift-equivariant toy 'model': smooths, then scales."""
    def __call__(self, x):
        k = torch.ones(1, 1, 3, 3) / 9.0
        c, t, h, w = x.shape
        y = F.conv2d(F.pad(x.reshape(c * t, 1, h, w), (1, 1, 1, 1), mode="replicate"), k)
        return 0.6 * y.reshape(c, t, h, w)


def _run_euler(sched, x, model, move=None, gen=None):
    """comfy's euler with the wrapper's protocol around the model calls."""
    n = len(sched) - 1
    first = move.first_step(n) if move else n
    for i in range(n):
        s, s_next = float(sched[i]), float(sched[i + 1])
        if i >= first:
            xm = cm.warp(x, move, s, gen)
            d = model(xm)
            if i < n - 1:
                d = cm.unwarp(d, x, move)
        else:
            d = model(x)
        x = d if s_next == 0 else d + (s_next / s) * (x - d)
    return x


def test_a_shift_equivariant_model_gives_the_plain_result_panned():
    sched = [1.0, 0.973, 0.923, 0.8, 0.0]
    x0 = torch.randn(2, 9, 12, 20, generator=_gen(5))
    move = cm.Move(pan_x=0.5, step=3)                    # takes hold at step 3 of 4
    plain = _run_euler(sched, x0.clone(), _Blur())
    moved = _run_euler(sched, x0.clone(), _Blur(), move, _gen())
    shifted = cm.warp(plain, move, 0.0, _gen())
    inner = (slice(None), slice(None), slice(2, 10), slice(3, 8))     # clear of the blur reaching the edges
    assert torch.allclose(moved[inner], shifted[inner], atol=1e-5)
    assert not torch.allclose(moved, plain)


def test_a_move_starting_after_the_last_step_or_no_move_changes_nothing():
    assert cm.Move(pan_x=0.2, step=3).first_step(4) == 2      # 1-based step 3 = index 2
    assert cm.Move(step=9).first_step(4) == 3 and cm.Move(step=1).first_step(4) == 0
    assert cm.Move(step=0).first_step(4) == 0 and cm.Move(step=5).first_step(1) == 0
    assert cm.Move().still() and not cm.Move(zoom=1.2).still()
    assert cm.clamp_pan(7) == 1.0 and cm.clamp_pan(float("nan")) == 0.0


def test_describe_reads_back_the_move():
    m = cm.Move(pan_x=-0.3, zoom=1.5, focus_x=0.2, step=3)
    assert m.describe(4) == "pan left 0.30, zoom in x1.50 toward (0.20, 0.50), from step 3 of 4"


# ── the installed wrapper, end to end on a packed picture+sound latent ──────────────────

class _Net:
    def __init__(self, shapes):
        self.latent_shapes = shapes


class _Patcher:
    def __init__(self, shapes):
        self.model = type("M", (), {})()
        self.model.latent_shapes = shapes
        self.model_options = {}

    def clone(self):
        c = _Patcher(self.model.latent_shapes)
        c.model_options = dict(self.model_options)
        return c


def _packed_run(move, sched, video, audio, seed=1):
    import samplers
    shapes = [tuple(video.shape[None:] and (1, *video.shape)), (1, *audio.shape)]
    patcher = _Patcher(shapes)
    stats = {}
    if move is not None:
        patcher, stats = samplers.FunPackLTXAVSceneChainSampler()._install_camera_move(patcher, move, seed)
    wrapper = patcher.model_options.get("model_function_wrapper")
    blur = _Blur()
    nv = video.numel()

    def apply_fn(x, t, **c):
        v = blur(x[0, 0, :nv].reshape(video.shape)).reshape(1, 1, nv)
        return torch.cat([v, 0.6 * x[..., nv:]], dim=-1)

    x = torch.cat([video.reshape(1, 1, nv), audio.reshape(1, 1, -1)], dim=-1)
    n = len(sched) - 1
    for i in range(n):
        s, s_next = float(sched[i]), float(sched[i + 1])
        args = {"input": x, "timestep": torch.tensor([s]), "cond_or_uncond": [0],
                "c": {"transformer_options": {"sample_sigmas": torch.tensor(sched),
                                              "sigmas": torch.tensor([s])}}}
        d = wrapper(apply_fn, args) if wrapper else apply_fn(x, None)
        x = d if s_next == 0 else d + (s_next / s) * (x - d)
    return x[0, 0, :nv].reshape(video.shape), x[0, 0, nv:], stats


def test_the_installed_wrapper_moves_the_picture_and_never_the_sound():
    sched = [1.0, 0.973, 0.923, 0.8, 0.0]
    video = torch.randn(2, 9, 12, 20, generator=_gen(6))
    audio = torch.randn(2, 30, generator=_gen(7))
    plain_v, plain_a, _ = _packed_run(None, sched, video, audio)
    move = cm.Move(pan_x=0.5, step=3)
    moved_v, moved_a, stats = _packed_run(move, sched, video, audio)
    assert stats["moved"] == 2                                   # steps 3 and 4 of 4
    assert torch.allclose(moved_a, plain_a)                      # the sound's path is unchanged
    inner = (slice(None), slice(None), slice(2, 10), slice(3, 8))
    assert torch.allclose(moved_v[inner], cm.warp(plain_v, move, 0.0, _gen())[inner], atol=1e-5)
    assert not torch.allclose(moved_v, plain_v)


def test_the_wrapper_stands_down_and_says_why_on_batches_and_context_windows():
    import samplers
    patcher = _Patcher([(1, 2, 9, 12, 20), (1, 60)])
    patched, stats = samplers.FunPackLTXAVSceneChainSampler()._install_camera_move(
        patcher, cm.Move(pan_x=0.5, step=1), 1)
    wrapper = patched.model_options["model_function_wrapper"]
    sched = torch.tensor([1.0, 0.9, 0.0])
    x = torch.randn(2, 1, 2 * 9 * 12 * 20 + 60)                  # a batched call
    out = wrapper(lambda inp, t, **c: inp * 0.5,
                  {"input": x, "timestep": torch.tensor([1.0]),
                   "c": {"transformer_options": {"sample_sigmas": sched, "sigmas": torch.tensor([1.0])}}})
    assert torch.equal(out, x * 0.5) and "batched" in stats["why"] and stats["moved"] == 0
    x1 = x[:1]
    wrapper(lambda inp, t, **c: inp * 0.5,
            {"input": x1, "timestep": torch.tensor([1.0]),
             "c": {"transformer_options": {"sample_sigmas": sched, "sigmas": torch.tensor([1.0]),
                                            "context_window": object()}}})
    assert stats["why"] == "context windows" and stats["moved"] == 0


def test_the_sampler_inputs_agree_with_the_editor_defaults():
    import samplers
    from movie_editor.backend import settings_card
    opt = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()["optional"]
    for name in ["camera_move", "camera_pan_x", "camera_pan_y", "camera_zoom",
                 "camera_focus_x", "camera_focus_y", "camera_step"]:
        assert opt[name][1]["default"] == settings_card._EDITOR_DEFAULTS[name], name
    assert opt["camera_move"][1]["default"] == "off"
