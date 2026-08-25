"""Per-modality AdaLN gain: per-block write strength for video / prompt / audio.

Every H3 DiT block carries `AdalnProj(t_dim, hidden, expand=6, modalities=3)` — shift,
scale and gate for attention and for the MLP, in three copies, one per modality. Rows are
indexed `t_row * 3 + tag` (video 0, text 1, audio 2), so one modality's rows across every
timestep are the slice `[tag::3]`. The GATE scales what a block writes back into a row
range, which makes it per-block, per-stream write gain.

This is a sampler-side visual-behaviour op. It reads no refinement key, no rating and no
learned direction, so it is unaffected by conditioning steering being off.
"""
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minimax_h3 as h3


class FakeProj(torch.nn.Module):
    """Stands in for AdalnProj: returns `expand` tensors of [M*modalities, hidden]."""

    def __init__(self, modalities=3, expand=6, timesteps=2, hidden=4):
        super().__init__()
        self.modalities = modalities
        self.expand = expand
        rows = timesteps * modalities
        # row value = its own index, so a scaled row is trivially identifiable
        self.buf = torch.arange(1, rows * expand * hidden + 1,
                                dtype=torch.float32).view(rows, expand * hidden)

    def forward(self, t_emb):
        return self.buf.chunk(self.expand, dim=-1)


def gates(out):
    return out[2], out[5]


def test_a_gain_scales_only_its_own_modality():
    inner = FakeProj()
    wrapped = h3.AdalnModalityGain(inner, {h3.MODALITY_TAGS["audio"]: 0.5})
    base_msa, base_mlp = gates(inner(None))
    msa, mlp = gates(wrapped(None))
    for tag in (0, 1):
        assert torch.equal(msa[tag::3], base_msa[tag::3])
        assert torch.equal(mlp[tag::3], base_mlp[tag::3])
    assert torch.allclose(msa[2::3], base_msa[2::3] * 0.5)
    assert torch.allclose(mlp[2::3], base_mlp[2::3] * 0.5)


def test_only_the_gate_chunks_move():
    """shift and scale modulate what a block READS; the gate is what it WRITES."""
    inner = FakeProj()
    wrapped = h3.AdalnModalityGain(inner, {0: 0.25})
    base, out = inner(None), wrapped(None)
    for idx in (0, 1, 3, 4):
        assert torch.equal(out[idx], base[idx])


def test_every_timestep_row_of_that_modality_is_scaled():
    """Rows are t_row*3+tag, so a modality appears once per unique timestep."""
    inner = FakeProj(timesteps=3)
    wrapped = h3.AdalnModalityGain(inner, {1: 2.0})
    base_msa, _ = gates(inner(None))
    msa, _ = gates(wrapped(None))
    assert msa[1::3].shape[0] == 3
    assert torch.allclose(msa[1::3], base_msa[1::3] * 2.0)


def test_the_source_buffer_is_not_mutated():
    """chunk() returns views onto one tensor — scaling in place would corrupt the module."""
    inner = FakeProj()
    before = inner.buf.clone()
    h3.AdalnModalityGain(inner, {0: 0.1})(None)
    h3.AdalnModalityGain(inner, {0: 0.1})(None)
    assert torch.equal(inner.buf, before)


def test_gain_of_one_is_dropped_entirely():
    wrapped = h3.AdalnModalityGain(FakeProj(), {0: 1.0, 1: 1.0})
    assert wrapped.gains == {}


def test_a_single_modality_projection_is_left_alone():
    """FinalLayer uses AdalnProj too, with modalities=1 — different row meaning."""
    inner = FakeProj(modalities=1, expand=2)
    wrapped = h3.AdalnModalityGain(inner, {0: 0.5})
    assert all(torch.equal(a, b) for a, b in zip(wrapped(None), inner(None)))


# ── attaching it to a model ─────────────────────────────────────────────────

class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adaln_proj = FakeProj()


class FakePatcher:
    def __init__(self, n=4):
        self.objects = {"diffusion_model.blocks": [FakeBlock() for _ in range(n)]}
        for i in range(n):
            self.objects[f"diffusion_model.blocks.{i}.adaln_proj"] = \
                self.objects["diffusion_model.blocks"][i].adaln_proj
        self.patched = {}

    def get_model_object(self, name):
        if name in self.patched:
            return self.patched[name]
        return self.objects[name]

    def clone(self):
        other = FakePatcher.__new__(FakePatcher)
        other.objects = self.objects
        other.patched = dict(self.patched)
        return other

    def add_object_patch(self, name, obj):
        self.patched[name] = obj


def test_all_ones_returns_the_same_model_unpatched():
    model = FakePatcher()
    out, note = h3.apply_adaln_gains(model, {"video": 1.0, "text": 1.0, "audio": 1.0})
    assert out is model and note is None
    assert model.patched == {}


def test_every_block_is_wrapped():
    model = FakePatcher(n=4)
    out, note = h3.apply_adaln_gains(model, {"video": 0.8})
    assert out is not model
    assert len(out.patched) == 4
    assert all(isinstance(v, h3.AdalnModalityGain) for v in out.patched.values())
    assert "4 blocks" in note


def test_it_says_the_anchor_moves_with_the_video():
    """Pins and reference images carry the VIDEO tag — a real limit, stated not hidden."""
    _out, note = h3.apply_adaln_gains(FakePatcher(), {"video": 0.8})
    assert "reference images ride the VIDEO tag" in note


def test_object_patches_are_used_not_method_replacement():
    """ComfyUI restores object patches on unpatch. Degradation outliving a reset has been a
    real bug here; this is the mechanism that stops it."""
    src = inspect.getsource(h3.apply_adaln_gains)
    assert "add_object_patch" in src
    assert "model.clone()" in src


def test_an_unknown_model_declines_with_a_reason():
    class NoBlocks:
        def get_model_object(self, name):
            raise KeyError(name)
    out, note = h3.apply_adaln_gains(NoBlocks(), {"video": 0.5})
    assert isinstance(out, NoBlocks)
    assert "no DiT blocks" in note


def test_unknown_modality_names_are_ignored():
    model = FakePatcher()
    out, note = h3.apply_adaln_gains(model, {"nonsense": 0.5})
    assert out is model and note is None


# ── independence from the refinement path ──────────────────────────────────

def test_the_installer_reads_no_rating_or_key():
    """Explicitly separate from conditioning steering: turning value guidance off must not
    turn this off, and this must work with no Refiner in the graph at all."""
    import samplers
    fn = samplers.FunPackLTXAVSceneChainSampler._install_h3_adaln_gains
    src = inspect.getsource(fn)
    # The prose says what it does NOT read, so check the code, not the docstring.
    doc = inspect.getdoc(fn) or ""
    for line in doc.splitlines():
        src = src.replace(line, "")
    for forbidden in ("refinement_key", "value_fn", "phrase_memory",
                      "global_state", "conditioning_deltas", "refine_v2"):
        assert forbidden not in src, f"{forbidden} leaked into a sampler-side op"


def test_the_installer_takes_the_model_and_the_conditioning_only():
    """The conditioning is the bridge: Studio tags the learned gains onto it. The sampler
    never opens a refinement key — that is the Studio/Sampler boundary."""
    import samplers
    params = inspect.signature(
        samplers.FunPackLTXAVSceneChainSampler._install_h3_adaln_gains).parameters
    assert list(params) == ["self", "model", "positive"]


def _sampler(mode="learned", **widgets):
    import samplers
    node = samplers.FunPackLTXAVSceneChainSampler.__new__(
        samplers.FunPackLTXAVSceneChainSampler)
    node._h3_gain_mode = mode
    for key, value in {"video": 1.0, "prompt": 1.0, "audio": 1.0}.items():
        setattr(node, f"_h3_gain_{key}", widgets.get(key, value))
    node._h3_prompt_scale = widgets.get("prompt_scale", 1.0)
    return node


def _tagged(gains):
    return [[torch.zeros(1, 4, 8), {"funpack_h3_gains": gains}]]


def test_learned_gains_win_over_the_widgets():
    node = _sampler(video=1.3, prompt=0.7)
    out = node._h3_render_gains(_tagged({"video": 0.9, "prompt": 1.1,
                                         "audio": 1.0, "prompt_scale": 1.2}))
    assert out["video"] == pytest.approx(0.9)
    assert out["prompt"] == pytest.approx(1.1)
    assert out["prompt_scale"] == pytest.approx(1.2)


def test_manual_mode_uses_the_widgets_and_ignores_the_key():
    node = _sampler(mode="manual", video=1.3)
    out = node._h3_render_gains(_tagged({"video": 0.5}))
    assert out["video"] == pytest.approx(1.3)


def test_manual_mode_works_with_no_conditioning_at_all():
    """The explicit override has to survive a graph with no Refiner in it."""
    assert _sampler(mode="manual", video=0.8)._h3_render_gains(None)["video"] == pytest.approx(0.8)


def test_learned_mode_before_any_rating_renders_at_trained_strength():
    """Not the widgets: in learned mode those are not what the user is steering with, and
    silently honouring them would make an unrated run look like a learned one."""
    node = _sampler(video=1.3, prompt=0.6)
    out = node._h3_render_gains([[torch.zeros(1, 4, 8), {}]])
    assert all(value == 1.0 for value in out.values())


def test_a_missing_key_in_the_learned_dict_falls_back_to_neutral():
    out = _sampler()._h3_render_gains(_tagged({"video": 0.9}))
    assert out["video"] == pytest.approx(0.9)
    assert out["audio"] == 1.0 and out["prompt_scale"] == 1.0


def test_learned_is_the_default_mode():
    import samplers
    spec = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()
    fields = {**spec.get("required", {}), **spec.get("optional", {})}
    assert fields["h3_gain_mode"][0] == ["learned", "manual"]
    assert fields["h3_gain_mode"][1]["default"] == "learned"


def test_the_widgets_exist_and_default_to_untouched():
    import samplers
    spec = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()
    fields = {**spec.get("required", {}), **spec.get("optional", {})}
    for name in ("h3_gain_video", "h3_gain_prompt", "h3_gain_audio"):
        assert name in fields, name
        assert fields[name][1]["default"] == 1.0


def test_sample_accepts_them_with_defaults():
    import samplers
    params = inspect.signature(samplers.FunPackLTXAVSceneChainSampler.sample).parameters
    for name in ("h3_gain_video", "h3_gain_prompt", "h3_gain_audio"):
        assert params[name].default == 1.0
