"""Tests for built-in pipeline dependency detection."""
from movie_editor.backend import pipeline_deps


def test_no_missing_when_all_core_present():
    oi = {cls: {} for cls in pipeline_deps.required_core_classes()}
    assert pipeline_deps.missing_core_classes(oi) == []
    assert pipeline_deps.missing_packs(oi) == []
    st = pipeline_deps.status_payload(oi, manager_available=True)
    assert st["needs_install"] is False


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
    st = pipeline_deps.status_payload(oi, manager_available=False)
    assert st["needs_install"] is True
    assert st["manager_available"] is False
    assert len(st["manual_urls"]) >= 2


def test_install_body_fallback_git():
    pack = pipeline_deps._PACK_BY_ID["ComfyUI-LTXVideo"]
    body = pipeline_deps._install_body(pack, None)
    assert body["version"] == "unknown"
    assert body["files"]
    assert "github.com" in body["files"][0]
