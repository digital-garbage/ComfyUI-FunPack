import pytest
import torch

from core import patching

KEY = "funpack.identity_transfer"


def _run(tiny, patched):
    patched.patch_model(device_to=torch.device("cpu"), load_weights=False)
    try:
        return tiny.run(patched)
    finally:
        patched.unpatch_model(unpatch_weights=False)


def _patched(tiny, seg=2.0, ref=None):
    from modules.conditioning.identity_transfer import patch
    p = tiny.patcher.clone()
    torch.manual_seed(1)
    patch.install(p, ref if ref is not None else torch.randn(1, 128, 1, 2, 2), seg, KEY)
    return p


def test_the_reference_goes_in_and_is_cut_off_again_so_shapes_are_unchanged(tiny_ltx):
    base_v, base_a = tiny_ltx.run()
    v, a = _run(tiny_ltx, _patched(tiny_ltx))
    assert v.shape == base_v.shape and a.shape == base_a.shape
    assert not torch.allclose(v, base_v)            # the reference tokens were attended to


def test_the_source_phase_tag_changes_the_result(tiny_ltx):
    ref = torch.randn(1, 128, 1, 2, 2)
    v2, _ = _run(tiny_ltx, _patched(tiny_ltx, seg=2.0, ref=ref))
    v0, _ = _run(tiny_ltx, _patched(tiny_ltx, seg=0.0, ref=ref))
    assert not torch.allclose(v2, v0)


def test_it_comes_off_again(tiny_ltx):
    base_v, _ = tiny_ltx.run()
    p = _patched(tiny_ltx)
    assert patching.strip(p, KEY) == 4 and not p.object_patches
    v, _ = _run(tiny_ltx, p)
    assert torch.allclose(v, base_v)


def test_a_model_without_the_stages_is_refused_in_words(tiny_ltx):
    from modules.conditioning.identity_transfer import patch

    class NotLtx:
        def get_model_object(self, name):
            return None
    with pytest.raises(RuntimeError, match="LTX audio\\+video"):
        patch.install(NotLtx(), torch.zeros(1), 2.0, KEY)


def test_a_failing_stage_is_said_and_the_run_goes_on_without_the_reference(tiny_ltx, monkeypatch):
    from modules.conditioning.identity_transfer import patch
    said = []
    monkeypatch.setattr(patch.log, "once", lambda key, level, source, message: said.append(message))
    base_v, _ = tiny_ltx.run()
    v, _ = _run(tiny_ltx, _patched(tiny_ltx, ref=torch.randn(1, 5, 1, 2, 2)))     # wrong channel count
    assert said and "appending the reference tokens" in said[0]
    assert torch.allclose(v, base_v)


def test_the_node_passes_everything_through_with_no_image(tiny_ltx):
    from modules.conditioning.identity_transfer.nodes import FunPackIdentityTransfer
    pos, neg = [[torch.zeros(1, 2, 3), {}]], [[torch.zeros(1, 2, 3), {}]]
    out = FunPackIdentityTransfer.execute(tiny_ltx.patcher, None, pos, neg, {"samples": tiny_ltx.video}).result
    assert out[1] is pos and out[2] is neg and "untouched" in out[3] and not out[0].object_patches


def test_the_node_encodes_the_reference_at_the_latent_size_and_patches(tiny_ltx):
    from modules.conditioning.identity_transfer.nodes import FunPackIdentityTransfer
    seen = {}

    class Vae:
        downscale_index_formula = (8, 32, 32)

        def encode(self, pixels):
            seen["shape"] = tuple(pixels.shape)
            return torch.randn(1, 128, 1, 2, 2)

    image = torch.rand(1, 100, 80, 3)
    pos, neg = [[torch.zeros(1, 2, 3), {}]], [[torch.zeros(1, 2, 3), {}]]
    patched, p, n, status = FunPackIdentityTransfer.execute(
        tiny_ltx.patcher, Vae(), pos, neg, {"samples": tiny_ltx.video}, image=image).result
    assert seen["shape"] == (1, 64, 64, 3)               # latent 2x2 at /32
    assert "reference tokens appended (source phase 2)" in status and len(patched.object_patches) == 4


def test_the_arcface_tokens_are_appended_to_both_conditionings():
    from modules.conditioning.identity_transfer.projector import append_context_tokens
    cond = [[torch.zeros(1, 2, 6), {"attention_mask": torch.ones(1, 2)}]]
    out = append_context_tokens(cond, torch.ones(1, 4, 8))          # wider tokens are cut to the context
    assert out[0][0].shape == (1, 6, 6) and out[0][1]["attention_mask"].shape == (1, 6)


def test_the_rotation_composes_onto_both_rope_layouts():
    from modules.conditioning.identity_transfer.rope import rotate_overlap_freqs
    cos, sin = torch.ones(1, 6, 4), torch.zeros(1, 6, 4)
    c2, s2, _ = rotate_overlap_freqs((cos, sin, "split"), 2, 1.0)
    assert torch.equal(c2[:, :4], cos[:, :4]) and not torch.equal(c2[:, 4:], cos[:, 4:])
    mat = torch.eye(2).reshape(1, 1, 1, 1, 2, 2).expand(1, 6, 1, 4, 2, 2).clone()
    m2, _ = rotate_overlap_freqs((mat, "split"), 2, 1.0)
    assert torch.equal(m2[:, :4], mat[:, :4]) and not torch.equal(m2[:, 4:], mat[:, 4:])
    assert rotate_overlap_freqs((cos, sin, "split"), 0, 1.0)[0] is cos


def _guided(tiny, patched):
    """The real forward with a guide frame: a denoise_mask with a hard-pinned frame, keyframe_idxs, per-token timestep."""
    patched.patch_model(device_to=torch.device("cpu"), load_weights=False)
    try:
        video, dmask = torch.randn(1, 128, 3, 2, 2), torch.ones(1, 1, 3, 2, 2)
        dmask[:, :, 0] = 0.0
        dmask[:, :, -1] = 0.0
        kf = torch.zeros(1, 3, 4, 2)
        kf[:, 0] = 5
        dm = patched.model.diffusion_model
        ts = dm.patchifier.patchify((dmask * 500.0)[:, :1])[0].reshape(1, -1)
        opts = {"sample_sigmas": torch.tensor([1.0, .5, 0.0]), "sigmas": torch.tensor([.5])}
        with torch.inference_mode():
            return dm([video, tiny.audio], (ts, torch.full((1, 3), 500.0)), tiny.context, frame_rate=25,
                      transformer_options=opts, denoise_mask=dmask, keyframe_idxs=kf)
    finally:
        patched.unpatch_model(unpatch_weights=False)


def test_a_latent_with_guide_frames_still_runs(tiny_ltx):
    base_v, _ = _guided(tiny_ltx, tiny_ltx.patcher.clone())
    v, _ = _guided(tiny_ltx, _patched(tiny_ltx))
    assert v.shape == base_v.shape and not torch.allclose(v, base_v)
