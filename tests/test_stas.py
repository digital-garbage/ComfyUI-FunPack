"""STAS (stas.py + samplers._install_stas): the right rows and channels are set to
alpha x peak x sign, only on the first steps, only on the picture rows."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stas  # noqa: E402


def test_target_rows_are_frame0_plus_each_frames_edges():
    rows = stas.target_rows(frames=3, per_frame=25, edge=0.08)       # k = 2
    expect = set(range(25)) | {25, 26, 48, 49, 50, 51, 73, 74}
    assert set(rows.tolist()) == expect


def test_only_massive_channels_are_found_and_steered():
    v = torch.randn(200, 128) * 0.1
    v[:, 5] = 20.0 * torch.sign(torch.randn(200))     # massive on every token
    v[3, 5] = -40.0                                    # its peak
    dims, peaks, _ = stas.ma_dims(v)
    assert dims.tolist() == [5] and float(peaks[0]) == 40.0
    before = v.clone()
    stas.steer(v, torch.tensor([0, 1, 2]), dims, peaks, 2.0)
    assert torch.equal(v[:3, 5], 80.0 * torch.sign(before[:3, 5]))
    assert torch.equal(v[3:], before[3:]) and torch.equal(v[:, :5], before[:, :5])


def test_first_steps():
    assert stas.first_steps(50) == 20 and stas.first_steps(7) == 3 and stas.first_steps(1) == 1


class _Net:
    patch_size = (1, 2, 2)

    def __init__(self):
        self.blocks = [lambda h: h] * 4


class _Model:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.model = type("M", (), {})()
        self.model.diffusion_model = _Net()

    def clone(self):
        m = _Model()
        m.model_options = {k: (dict(v) if isinstance(v, dict) else v)
                           for k, v in self.model_options.items()}
        return m


def _call(patched, h, step, total, video_rows):
    import samplers
    hook = patched.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 2)]
    sched = torch.linspace(1.0, 0.0, total + 1)
    args = {"img": h, "mod_segments": video_rows,
            "transformer_options": {"sample_sigmas": sched, "sigmas": sched[step:step + 1]}}
    return hook(args, {"original_block": lambda a: {"img": a["img"]}})["img"], samplers


def test_hook_steers_picture_rows_on_early_steps_only(monkeypatch):
    import samplers
    import h3_repr_steering as rs
    # 10 text rows, then 2 frames x (4x6 latent -> 2x3 = 6 tokens) = 12 video rows, then audio
    def mask(video_rows, seq_len, device):
        m = torch.zeros(seq_len, dtype=torch.bool)
        m[10:22] = True
        return m
    monkeypatch.setattr(rs, "video_mask_from_mod_segments", mask)
    samples = torch.zeros(1, 24, 2, 4, 6)
    patched, report = samplers.FunPackLTXAVSceneChainSampler()._install_stas(_Model(), 2, 2.0, samples)
    h = torch.randn(30, 64) * 0.1
    h[10:22, 3] = 30.0
    h[12, 3] = 50.0
    out, _ = _call(patched, h.clone(), step=0, total=7, video_rows=None)
    # every video row is in S here (frame 0 = all of frame 0; edges k=1 of 6 per frame)
    targets = [10, 11, 12, 13, 14, 15, 16, 21]
    assert torch.allclose(out[targets, 3], torch.full((8,), 100.0))
    assert torch.equal(out[:10], h[:10]) and torch.equal(out[22:], h[22:])      # text/audio
    assert torch.equal(out[17:21], h[17:21])                                    # interior
    late, _ = _call(patched, h.clone(), step=5, total=7, video_rows=None)
    assert torch.equal(late, h)                                                 # past 40%
    assert "channels 3" in report()[0] and report()[1] == 1


def test_unreadable_layout_is_declared(monkeypatch):
    import samplers
    import h3_repr_steering as rs
    monkeypatch.setattr(rs, "video_mask_from_mod_segments", lambda *a: None)
    patched, report = samplers.FunPackLTXAVSceneChainSampler()._install_stas(
        _Model(), 2, 2.0, torch.zeros(1, 24, 2, 4, 6))
    h = torch.randn(30, 8)
    out, _ = _call(patched, h.clone(), step=0, total=7, video_rows=None)
    assert torch.equal(out, h) and "nothing steered" in report()[0] and report()[1] == 0


def test_weak_branch_calls_are_steered_but_not_counted(monkeypatch):
    import samplers
    import h3_repr_steering as rs
    monkeypatch.setattr(rs, "video_mask_from_mod_segments",
                        lambda v, n, d: torch.arange(n) < 12)
    patched, report = samplers.FunPackLTXAVSceneChainSampler()._install_stas(
        _Model(), 2, 2.0, torch.zeros(1, 24, 2, 4, 6))
    hook = patched.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 2)]
    sched = torch.linspace(1.0, 0.0, 8)
    h = torch.randn(20, 64) * 0.1
    h[:12, 3] = 30.0
    to = {"sample_sigmas": sched, "sigmas": sched[:1], samplers.WEAK_BRANCH_FLAG: True}
    out = hook({"img": h.clone(), "mod_segments": None, "transformer_options": to},
               {"original_block": lambda a: {"img": a["img"]}})["img"]
    assert float(out[0, 3]) == 60.0 and report()[1] == 0
