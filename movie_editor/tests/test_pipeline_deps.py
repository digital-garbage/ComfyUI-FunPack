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
