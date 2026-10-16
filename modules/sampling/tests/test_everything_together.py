"""Every rating-driven and sampling modifier on at once: nothing is rejected or reordered into a cycle."""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def test_all_modifiers_install_together_in_a_sane_order(tiny_h3, tmp_path, monkeypatch):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    settings = {"taste": {"key": "fox"}}
    for m, extra in {"dynashift": {}, "output_guidance": {}, "trajectory_guidance": {}, "score_slider": {},
                     "decisiveness": {}, "shot_memory": {}, "late_branch": {"block": 2},
                     "stas": {"block": 1}, "camera_move": {"pan_x": 0.3},
                     "reins": {"blocks": "3"}}.items():
        settings[m] = {"enabled": True, **extra}
    patched, status = FunPackLoadModifiers.execute(tiny_h3.patcher, settings).result
    for m in ("decisiveness", "shot_memory", "late_branch", "stas", "camera_move", "block_influence"):
        assert m in status, (m, status)
    order = list(patched.wrappers[WrappersMP.APPLY_MODEL])
    assert order.index("funpack.camera_move") < order.index("funpack.late_branch")
    assert order.index("funpack.score_slider") < order.index("funpack.late_branch")
