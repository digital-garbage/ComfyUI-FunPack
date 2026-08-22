"""Family detection from a checkpoint's own header.

Fixtures, never real checkpoints: the signatures are key NAMES, so a synthetic safetensors
header exercises the same code path a 40 GB file would.
"""
import json
import struct

import pytest

from movie_editor.backend import model_probe


def write_safetensors(path, keys, metadata=None):
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00\x00")
    return path


H3 = ["video_patch_proj.weight", "audio_patch_proj.weight", "blocks.0.attn.qkv_proj.weight"]
LTXAV = ["adaln_single.emb.timestep_embedder.linear_1.bias", "audio_adaln_single.linear.weight"]
LTXV = ["adaln_single.emb.timestep_embedder.linear_1.bias"]


def test_h3_is_recognised_by_its_two_patch_projections():
    assert model_probe.detect_arch(H3) == "minimax_h3"


def test_ltx_with_an_audio_branch_is_ltxav():
    assert model_probe.detect_arch(LTXAV) == "ltxav"


def test_ltx_without_an_audio_branch_is_reported_as_video_only():
    assert model_probe.detect_arch(LTXV) == "ltxv"


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model.", "diffusion_model."])
def test_the_signature_is_found_behind_any_packing_prefix(prefix):
    assert model_probe.detect_arch([prefix + k for k in H3]) == "minimax_h3"


def test_an_unknown_architecture_answers_none_rather_than_defaulting_to_ltx():
    # The whole point: silence must not be read as "LTX". A wrong family wires the wrong
    # graph and shows up as a stray port, not as a family error.
    assert model_probe.detect_arch(["model.diffusion_model.some_wan_block.weight"]) is None
    assert model_probe.detect_arch([]) is None


def test_half_an_h3_signature_is_not_h3():
    assert model_probe.detect_arch(["video_patch_proj.weight"]) is None


def test_a_real_header_round_trips(tmp_path):
    p = write_safetensors(tmp_path / "minimax_h3_fl2va_bf16.safetensors", H3)
    out = model_probe.detect_family(p)
    assert out["family"] == "minimax_h3"
    assert out["detected"] is True
    assert "MiniMax H3" in out["reason"]


def test_a_video_only_ltx_wires_ltxav_but_says_the_audio_branch_is_empty(tmp_path):
    p = write_safetensors(tmp_path / "ltxv.safetensors", LTXV)
    out = model_probe.detect_family(p)
    assert out["family"] == "ltxav"      # no ltxv wiring exists; ltxav is the graph
    assert out["arch"] == "ltxv"         # but the caller is told which it really was
    assert "audio branch" in out["reason"]


def test_metadata_is_not_mistaken_for_a_tensor(tmp_path):
    p = write_safetensors(tmp_path / "m.safetensors", H3, metadata={"format": "pt"})
    assert model_probe.detect_family(p)["family"] == "minimax_h3"


def test_an_unknown_checkpoint_reports_why_instead_of_guessing(tmp_path):
    p = write_safetensors(tmp_path / "wan2.2.safetensors", ["blocks.0.self_attn.q.weight"])
    out = model_probe.detect_family(p)
    assert out["family"] is None
    assert out["detected"] is False
    assert "no wiring" in out["reason"] or "no LTX or MiniMax H3 signature" in out["reason"]


def test_a_pickle_checkpoint_is_declined_rather_than_executed(tmp_path):
    p = tmp_path / "model.ckpt"
    p.write_bytes(b"\x80\x04\x95 pickled")
    out = model_probe.detect_family(p)
    assert out["family"] is None
    assert ".safetensors" in out["reason"]


def test_a_truncated_or_junk_file_does_not_raise(tmp_path):
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"\x00\x01\x02")
    assert model_probe.detect_family(junk)["detected"] is False
    absurd = tmp_path / "absurd.safetensors"
    absurd.write_bytes(struct.pack("<Q", 2**60) + b"{}")
    assert model_probe.read_safetensors_keys(absurd) is None


def test_a_missing_file_is_reported_not_raised(tmp_path):
    out = model_probe.detect_family(tmp_path / "nope.safetensors")
    assert out["detected"] is False and "not found" in out["reason"]


# --- reading the family out of a pipeline config ---------------------------------

def slot(cls="FunPackDiffusionModelLoader", role="unet", **inputs):
    return {"id": "fp_unet", "role": role, "node_class": cls, "inputs": dict(inputs)}


def test_the_diffusion_slot_is_found_by_role():
    models = {"slots": [slot(model_name="minimax_h3_fl2va_bf16.safetensors")]}
    assert model_probe.diffusion_model_file(models) == "minimax_h3_fl2va_bf16.safetensors"


def test_an_imported_workflows_loader_is_found_by_class_and_widget_name():
    # A workflow import brings whatever loader its author used, with its own widget name.
    models = {"slots": [slot(cls="UNETLoader", role="custom", unet_name="ltxav.safetensors")]}
    assert model_probe.diffusion_model_file(models) == "ltxav.safetensors"


def test_a_config_with_no_diffusion_model_says_so_rather_than_guessing():
    out = model_probe.probe_models({"slots": []})
    assert out["family"] is None and "no diffusion model" in out["reason"]


def test_probe_reports_the_family_for_a_real_slot(tmp_path):
    f = write_safetensors(tmp_path / "h3.safetensors", H3)
    models = {"slots": [slot(model_name="h3.safetensors")]}
    out = model_probe.probe_models(models, resolve=lambda name: str(f))
    assert out["family"] == "minimax_h3"
    assert out["file"] == "h3.safetensors"


def test_a_file_that_is_not_in_the_models_folders_is_named(tmp_path):
    models = {"slots": [slot(model_name="ghost.safetensors")]}
    out = model_probe.probe_models(models, resolve=lambda name: None)
    assert out["detected"] is False
    assert "ghost.safetensors" in out["reason"]


def test_an_unidentifiable_checkpoint_proposes_no_family(tmp_path):
    # The caller keeps whatever family was already set — the point of the whole change is
    # that nothing silently becomes LTX.
    f = write_safetensors(tmp_path / "wan.safetensors", ["blocks.0.self_attn.q.weight"])
    out = model_probe.probe_models({"slots": [slot(model_name="wan.safetensors")]},
                                   resolve=lambda name: str(f))
    assert out["family"] is None


def test_gguf_without_the_library_says_so_and_proposes_no_family(tmp_path, monkeypatch):
    """A .gguf that cannot be read must never fall through to a guess — the previous family
    stands, and the panel says why plus where to set it by hand."""
    import movie_editor.backend.model_probe as mp
    p = tmp_path / "ltx-Q4_K_M.gguf"
    p.write_bytes(b"GGUF\x03\x00\x00\x00")
    monkeypatch.setattr(mp, "read_gguf_keys", lambda _p: None)
    out = mp.detect_family(p)
    assert out["family"] is None
    assert out["detected"] is False
    assert "gguf" in out["reason"].lower()
    assert "Model family" in out["reason"]


def test_gguf_is_identified_from_the_same_signatures(tmp_path, monkeypatch):
    """The architecture signatures are KEY NAMES, so the container they came out of is
    irrelevant — a GGUF and a safetensors of the same model must agree."""
    import movie_editor.backend.model_probe as mp
    p = tmp_path / "h3-Q8_0.gguf"
    p.write_bytes(b"GGUF")
    monkeypatch.setattr(mp, "read_gguf_keys",
                        lambda _p: ["video_patch_proj.weight", "audio_patch_proj.weight"])
    out = mp.detect_family(p)
    assert out["arch"] == "minimax_h3"
    assert out["family"] == "minimax_h3"
    assert out["detected"] is True


def test_gguf_with_no_known_signature_proposes_nothing(tmp_path, monkeypatch):
    import movie_editor.backend.model_probe as mp
    p = tmp_path / "something.gguf"
    p.write_bytes(b"GGUF")
    monkeypatch.setattr(mp, "read_gguf_keys", lambda _p: ["blk.0.attn_q.weight"])
    out = mp.detect_family(p)
    assert out["family"] is None
    assert "no LTX or MiniMax H3 signature" in out["reason"]


def test_probe_reads_a_gguf_renamed_to_safetensors(tmp_path, monkeypatch):
    """Detection follows the container, not the filename — the loader does the same, so the
    two cannot disagree about what a file is."""
    import movie_editor.backend.model_probe as mp
    p = tmp_path / "h3-Q8_0.safetensors"
    p.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x80" * 64)
    monkeypatch.setattr(mp, "read_gguf_keys",
                        lambda _p: ["video_patch_proj.weight", "audio_patch_proj.weight"])
    out = mp.detect_family(p)
    assert out["arch"] == "minimax_h3"
