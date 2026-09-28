"""Late-branch guidance (late_guidance.py + samplers._install_late_branch): the weak pass
reuses the shared blocks instead of running them, skips the branch block, runs the tail,
keeps capture hooks out of it, and the picture is pushed away from it."""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import late_guidance as lg  # noqa: E402


class _Blocks:
    """A 6-block stand-in for H3's DiT: block i adds (i+1) into its input in place."""
    def __init__(self):
        self.calls = []
        self.blocks = [self._make(i) for i in range(6)]

    def _make(self, i):
        def block(h):
            self.calls.append(i)
            return h.add_(float(i + 1))
        return block


class _Model:
    def __init__(self, net):
        self.model_options = {"transformer_options": {}}
        self.model = type("M", (), {})()
        self.model.diffusion_model = net

    def clone(self):
        m = _Model(self.model.diffusion_model)
        m.model_options = {k: (dict(v) if isinstance(v, dict) else v)
                           for k, v in self.model_options.items()}
        return m


def _apply(net):
    """What comfy's apply_model -> H3 forward does with dit patches, minus the maths."""
    def apply_fn(x, t, transformer_options=None, **_c):
        to = transformer_options or {}
        dit = (to.get("patches_replace") or {}).get("dit", {})
        h = x.clone()
        for i, block in enumerate(net.blocks):
            if ("double_block", i) in dit:
                h = dit[("double_block", i)](
                    {"img": h, "transformer_options": to},
                    {"original_block": lambda a, b=block: {"img": b(a["img"])}})["img"]
            else:
                h = block(h)
        return h
    return apply_fn


def _run(model, net, x):
    wrapper = model.model_options["model_function_wrapper"]
    return wrapper(_apply(net), {"input": x, "timestep": torch.tensor([0.5]),
                                 "c": {"transformer_options": model.model_options["transformer_options"]}})


def test_weak_pass_shares_the_head_skips_the_branch_and_pushes_away():
    import samplers
    net = _Blocks()
    seen = []

    def spy(args, extra):          # an existing hook in the tail, e.g. REINS capture
        seen.append(samplers._in_weak_branch(args))
        return extra["original_block"](args)
    base = _Model(net)
    base.model_options["transformer_options"] = {"patches_replace": {"dit": {("double_block", 5): spy}}}
    patched = samplers.FunPackLTXAVSceneChainSampler()._install_late_branch(base, 3, 0.5)[0]
    assert patched is not base

    x = torch.zeros(1, 2, 1, 2, 2)
    out = _run(patched, net, x)
    # normal: 0..5 all run = 21; weak: head reused (1+2+3=6), block 3 skipped, 4+5 run -> 17
    assert net.calls == [0, 1, 2, 3, 4, 5, 4, 5]
    assert torch.allclose(out, torch.full_like(x, 21 + 0.5 * (21 - 17)))
    assert seen == [False, True]      # the tail hook can tell the weak pass apart


def test_refuses_a_block_the_model_does_not_have():
    import samplers
    base = _Model(_Blocks())
    s = samplers.FunPackLTXAVSceneChainSampler()
    assert s._install_late_branch(base, 0, 0.5)[0] is base
    assert s._install_late_branch(base, 6, 0.5)[0] is base


def test_unreadable_packed_latent_is_left_alone(monkeypatch):
    import samplers
    monkeypatch.setattr(samplers, "_video_span", lambda model, x: None)
    net = _Blocks()
    patched, stats = samplers.FunPackLTXAVSceneChainSampler()._install_late_branch(_Model(net), 3, 0.5)
    x = torch.zeros(1, 1, 8)
    assert torch.allclose(_run(patched, net, x), torch.full_like(x, 21.0))
    assert stats["guided"] == 0 and "nothing guided" in stats["why"]   # said, not silent


def test_video_span_only_the_picture_is_guided(monkeypatch):
    import samplers
    monkeypatch.setattr(samplers, "_video_span", lambda model, x: (0, 5, (1, 1, 1, 1, 5)))
    net = _Blocks()
    patched = samplers.FunPackLTXAVSceneChainSampler()._install_late_branch(_Model(net), 3, 1.0)[0]
    out = _run(patched, net, torch.zeros(1, 1, 8))
    assert torch.allclose(out[..., :5], torch.full((1, 1, 5), 25.0))   # 21 + 1*(21-17)
    assert torch.allclose(out[..., 5:], torch.full((1, 1, 3), 21.0))   # sound: normal pass


def test_negative_prompt_calls_are_not_guided():
    """CFG above 1: the negative call runs once, plain; the prompt call is still guided."""
    import samplers
    net = _Blocks()
    patched = samplers.FunPackLTXAVSceneChainSampler()._install_late_branch(_Model(net), 3, 0.5)[0]
    wrapper = patched.model_options["model_function_wrapper"]
    x = torch.zeros(1, 2, 1, 2, 2)
    c = {"transformer_options": patched.model_options["transformer_options"]}
    neg = wrapper(_apply(net), {"input": x, "timestep": torch.tensor([0.5]), "c": c,
                                "cond_or_uncond": [1]})
    assert net.calls == [0, 1, 2, 3, 4, 5] and torch.allclose(neg, torch.full_like(x, 21.0))
    pos = wrapper(_apply(net), {"input": x, "timestep": torch.tensor([0.5]), "c": c,
                                "cond_or_uncond": [0]})
    assert torch.allclose(pos, torch.full_like(x, 23.0))


def test_strength_learns_from_ratings(monkeypatch, tmp_path):
    d = lg.DIAL
    monkeypatch.setattr(d, "path", lambda key: str(tmp_path / f"{key}.late.json"))
    rng = random.Random(1)
    for _ in range(30):
        w, _note = d.choose("k", "learned", 0.5, rng=rng)
        d.save_pending("k", w)
        d.commit("k", 1.0 if w > 0.8 else -1.0)
    assert d.learned(d.read("k")["history"]) > 0.8
    assert d.commit("k", 1.0) is None                       # nothing pending
    d.save_pending("k", 0.3)
    assert d.commit("k", 0.0) is None                       # neutral: teaches nothing
    d.save_pending("k", 0.3)
    d.clear_pending("k")                                    # a run with it off
    assert d.commit("k", -1.0) is None
    assert d.choose("k", "off", 0.5)[0] is None
    assert d.choose("", "learned", 0.5)[0] is None          # no key: off


def test_installed_copies_free_without_a_full_gc():
    """A hook that holds its own ModelPatcher is a cycle: only a full gc frees it, and that
    clears ComfyUI's weak link to the parent copy first -> "memory leak with model".
    Real ComfyUI in a subprocess: this suite stubs `comfy`."""
    import os
    import subprocess
    import pytest
    root = Path(os.environ.get("COMFYUI_DIR", Path.home() / "Documents" / "ComfyUI"))
    if not (root / "comfy" / "model_patcher.py").exists():
        pytest.skip("no ComfyUI checkout to test against")
    code = f"""
import sys, gc, weakref
sys.path[:0] = [{str(Path(__file__).resolve().parents[1])!r}, {str(root)!r}]
import torch, comfy.model_patcher as mp, samplers
gc.disable()
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])
class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = Net()
root = mp.ModelPatcher(Base(), torch.device("cpu"), torch.device("cpu"))
s = samplers.FunPackLTXAVSceneChainSampler()
for name, install in (("late", lambda m: s._install_late_branch(m, 3, 0.5)[0]),
                      ("stas", lambda m: s._install_stas(m, 2, 2.0, torch.zeros(1, 24, 2, 4, 6))[0])):
    p = install(root.clone())
    ref = weakref.ref(p)
    del p
    assert ref() is None, name + " copy is in a reference cycle"
"""
    done = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
