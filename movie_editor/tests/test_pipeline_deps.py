"""Tests for built-in pipeline dependency detection."""
from pathlib import Path
from unittest.mock import patch

from movie_editor.backend import pipeline_deps


def test_no_missing_when_all_core_present():
    oi = {cls: {} for cls in pipeline_deps.required_core_classes()}
    assert pipeline_deps.missing_core_classes(oi) == []
    assert pipeline_deps.missing_packs(oi) == []
    st = pipeline_deps.status_payload(oi, manager_available=True)
    assert st["needs_install"] is False
    assert st["needs_manager_install"] is False


def test_missing_ltx_and_vhs_packs():
    oi = {
        "FunPackStudio": {},
        "FunPackLTXAVSceneChainSampler": {},
        "FunPackSaveRefinementLatent": {},
        "FunPackRefinementKeyLoader": {},
        "PrimitiveStringMultiline": {},
        "PrimitiveInt": {},
        "PrimitiveFloat": {},
    }
    missing = pipeline_deps.missing_core_classes(oi)
    assert "LTXVConditioning" in missing
    assert "VHS_VideoCombine" in missing
    packs = pipeline_deps.missing_packs(oi)
    ids = {p["id"] for p in packs}
    assert "ComfyUI-LTXVideo" in ids
    assert "comfyui-videohelpersuite" in ids
    st = pipeline_deps.status_payload(oi, manager_available=False, manager_on_disk=False)
    assert st["needs_install"] is True
    assert st["needs_manager_install"] is True
    assert st["needs_manager_restart"] is False
    assert len(st["manual_urls"]) >= 2


def test_manager_on_disk_needs_restart_not_install():
    oi = {
        "FunPackStudio": {},
        "FunPackLTXAVSceneChainSampler": {},
    }
    st = pipeline_deps.status_payload(oi, manager_available=False, manager_on_disk=True)
    assert st["needs_manager_install"] is False
    assert st["needs_manager_restart"] is True


def test_install_body_fallback_git():
    pack = pipeline_deps._PACK_BY_ID["ComfyUI-LTXVideo"]
    body = pipeline_deps._install_body(pack, None)
    assert body["version"] == "unknown"
    assert body["files"]
    assert "github.com" in body["files"][0]


def test_cancel_install_job():
    job = pipeline_deps.create_install_job(["ComfyUI-LTXVideo"])
    assert pipeline_deps.cancel_install_job(job["job_id"]) is True
    snap = pipeline_deps.get_install_job(job["job_id"])
    assert snap is not None


def test_install_manager_sync_skips_when_present(tmp_path):
    cn = tmp_path / "custom_nodes"
    mgr = cn / "ComfyUI-Manager"
    mgr.mkdir(parents=True)
    (mgr / ".git").mkdir()
    with patch.object(pipeline_deps, "custom_nodes_dir", return_value=cn):
        ok, msg = pipeline_deps.install_manager_sync()
    assert ok is True
    assert msg == "already_installed"


def test_custom_nodes_dir_uses_folder_paths(tmp_path, monkeypatch):
    cn = tmp_path / "custom_nodes"
    cn.mkdir()
    fp = type("FP", (), {"get_folder_paths": staticmethod(lambda _: [str(cn)])})()
    monkeypatch.setitem(__import__("sys").modules, "folder_paths", fp)
    assert pipeline_deps.custom_nodes_dir() == cn


# ── per-family setup ──────────────────────────────────────────────────────────
# Both families are real, selectable pipelines. H3 shipped in ComfyUI v0.30.0 with weights
# on Comfy-Org/MiniMax-H3, and it ships as TWO diffusion checkpoints (fl2va / ref2va) that
# load interchangeably but condition differently — the setup panel is the only place that
# can warn about that before a generation is spent, so these pin that it says so.

def test_both_families_are_offered_and_released():
    from movie_editor.backend import pipeline_deps as pd
    fams = {f["key"]: f for f in pd.families_payload()}
    assert set(fams) == {"ltxav", "minimax_h3"}
    assert fams["ltxav"]["released"] is True
    assert fams["minimax_h3"]["released"] is True
    # the two-checkpoint split is the one thing a user cannot discover from the graph
    note = fams["minimax_h3"]["note"]
    assert "fl2va" in note and "ref2va" in note
    assert "MiniMax-H3" in fams["minimax_h3"]["source_url"]
    assert fams["ltxav"]["note"] is None


def test_h3_readiness_lists_the_nodes_and_model_files_it_needs():
    from movie_editor.backend import pipeline_deps as pd
    r = pd.family_readiness({}, "minimax_h3")
    assert r["released"] is True
    # the AV latent node is required; the sigma-shift node is optional and must not block
    missing = {n["class"] for n in r["missing_nodes"]}
    assert missing == {"EmptyMiniMaxH3LatentAV"}
    assert {m["role"] for m in r["models"]} == {"unet", "clip", "video_vae", "audio_vae"}
    assert all(m["folder"] for m in r["models"])
    # the hints name real files, including both diffusion variants
    unet_hint = next(m["hint"] for m in r["models"] if m["role"] == "unet")
    assert "minimax_h3_fl2va" in unet_hint and "minimax_h3_ref2va" in unet_hint

    # on an older ComfyUI without the node, it is listed as missing — no code change
    r2 = pd.family_readiness({"EmptyMiniMaxH3LatentAV": {}}, "minimax_h3")
    assert r2["missing_nodes"] == []


def test_a_family_missing_its_own_nodes_still_counts_as_needing_setup():
    """Otherwise the modal never opens and the project looks ready when it cannot generate."""
    from movie_editor.backend import pipeline_deps as pd
    full_oi = {cls: {} for cls in pd.required_core_classes("minimax_h3")}
    s = pd.status_payload(full_oi, manager_available=True, family="minimax_h3")
    assert s["missing_packs"] == []          # nothing for Manager to install
    assert s["needs_setup"] is True          # ... but the H3 latent node is not installed
    assert s["family"] == "minimax_h3"

    # with the H3 node present too, setup is genuinely done
    s2 = pd.status_payload({**full_oi, "EmptyMiniMaxH3LatentAV": {}},
                           manager_available=True, family="minimax_h3")
    assert s2["needs_setup"] is False


def test_h3_does_not_ask_for_the_ltx_only_core_packs():
    """H3's core drops LTXVConditioning / Concat / LTXVAudioVAEDecode, so a user on H3
    must not be told to install a pack for nodes their graph never emits."""
    from movie_editor.backend import pipeline_deps as pd
    oi = {cls: {} for cls in pd.required_core_classes("minimax_h3")}
    assert pd.missing_core_classes(oi, "minimax_h3") == []
    # the same object_info is NOT enough for LTXAV
    ltx_missing = pd.missing_core_classes(oi, "ltxav")
    assert "LTXVConditioning" in ltx_missing and "LTXVConcatAVLatent" in ltx_missing


def test_family_lookup_falls_back_instead_of_raising():
    from movie_editor.backend import pipeline_deps as pd
    assert pd.family_setup("hailuo-9000")["label"] == pd.family_setup("ltxav")["label"]
    assert pd.family_setup(None)["label"] == pd.family_setup("ltxav")["label"]


# ── no role may target a node its family does not build ─────────────────────
# The reported symptom ("another Audio VAE output to a node that isn't there") came from
# LTXAV's inherited rules pointing audio_vae at LTXVAudioVAEDecode, which H3's core drops.
# Fixing the family fixed the cause; these make the shape of the bug impossible.

def test_no_role_target_points_at_a_node_outside_its_family_core():
    from movie_editor.backend import builder, pipeline_wiring as pw

    for family in ("ltxav", "minimax_h3"):
        core, _links, _ports = builder.family_core(family)
        classes = set(core.values())
        for role, targets in pw._role_targets(family).items():
            for _type, _out, port in targets:
                assert port.split(".", 1)[0] in classes, f"{family}/{role} -> {port}"


def test_no_default_wire_points_at_a_node_outside_its_family_core():
    """Default wires are applied without the user asking, so a stale one is the likeliest
    way a phantom port gets written into a saved project."""
    from movie_editor.backend import builder, pipeline_wiring as pw

    for family in ("ltxav", "minimax_h3"):
        core, _links, _ports = builder.family_core(family)
        classes = set(core.values())
        for role, wires in pw._default_wires(family).items():
            for _type, wire in wires.items():
                if isinstance(wire, str) and wire.startswith("port:"):
                    cls = wire[len("port:"):].split(".", 1)[0]
                    assert cls in classes, f"{family}/{role} -> {wire}"


def test_h3_routes_the_audio_vae_to_its_own_decoder():
    """The positive half: H3 must still get the two real targets it needs."""
    from movie_editor.backend import pipeline_wiring as pw

    ports = [t[2] for t in pw._role_targets("minimax_h3")["audio_vae"]]
    assert "VAEDecodeAudio.vae" in ports
    assert "FunPackLTXAVSceneChainSampler.audio_vae" in ports
    assert not any(p.startswith("LTXVAudioVAEDecode") for p in ports)


def test_ltxav_is_unchanged_by_the_guard():
    from movie_editor.backend import pipeline_wiring as pw

    assert [t[2] for t in pw._role_targets("ltxav")["audio_vae"]] == ["LTXVAudioVAEDecode.audio_vae"]
    assert pw._default_wires("ltxav")["audio_vae"] == {"VAE": "port:LTXVAudioVAEDecode.audio_vae"}
