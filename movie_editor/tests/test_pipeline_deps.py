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


# ── per-family setup placeholders ─────────────────────────────────────────────
# MiniMax H3 is not released — no merged nodes, no published weights. The setup flow has
# to say that plainly rather than either hiding the family or offering an install that
# cannot succeed. These pin the shape the placeholders must keep, so shipping the real
# thing is a data edit (flip `released`, fill the hints) and not a code change.

def test_both_families_are_offered_and_h3_is_marked_unreleased():
    from movie_editor.backend import pipeline_deps as pd
    fams = {f["key"]: f for f in pd.families_payload()}
    assert set(fams) == {"ltxav", "minimax_h3"}
    assert fams["ltxav"]["released"] is True
    assert fams["minimax_h3"]["released"] is False
    # a user picking it must be told why nothing generates yet, and where it comes from
    assert "not available yet" in fams["minimax_h3"]["note"]
    assert "15224" in fams["minimax_h3"]["source_url"]
    assert fams["ltxav"]["note"] is None


def test_h3_readiness_lists_the_nodes_and_model_files_it_is_waiting_on():
    from movie_editor.backend import pipeline_deps as pd
    r = pd.family_readiness({}, "minimax_h3")
    assert r["released"] is False
    # the AV latent node is required; the sigma-shift node is optional and must not block
    missing = {n["class"] for n in r["missing_nodes"]}
    assert missing == {"EmptyMiniMaxH3LatentAV"}
    assert {m["role"] for m in r["models"]} == {"unet", "clip", "video_vae", "audio_vae"}
    assert all(m["folder"] for m in r["models"])

    # once ComfyUI ships the node, it stops being listed as missing — no code change
    r2 = pd.family_readiness({"EmptyMiniMaxH3LatentAV": {}}, "minimax_h3")
    assert r2["missing_nodes"] == []


def test_an_unreleased_family_still_counts_as_needing_setup():
    """Otherwise the modal never opens and the project looks ready when it cannot generate."""
    from movie_editor.backend import pipeline_deps as pd
    full_oi = {cls: {} for cls in pd.required_core_classes("minimax_h3")}
    s = pd.status_payload(full_oi, manager_available=True, family="minimax_h3")
    assert s["missing_packs"] == []          # nothing for Manager to install
    assert s["needs_setup"] is True          # ... but setup is not finished
    assert s["readiness"]["released"] is False
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
