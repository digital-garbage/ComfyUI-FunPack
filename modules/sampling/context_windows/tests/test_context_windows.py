import pytest

from core import patching

KEY = "funpack.context_windows"
ON = {"enabled": True, "length": 145, "overlap": 40, "schedule": "standard_uniform", "fuse": "pyramid",
      "freenoise": True, "retain_first": False}


def _patcher(tiny):
    return patching.GuardedPatcher(tiny.patcher.clone(), KEY, patching.Dropped())


def test_off_installs_nothing(tiny_ltx):
    from modules.sampling.context_windows import install
    p = _patcher(tiny_ltx)
    assert install(p, {**ON, "enabled": False}, KEY) is None
    assert "context_handler" not in p._patcher.model_options


def test_on_builds_cores_handler_in_latent_frames_with_both_wrappers(tiny_ltx, monkeypatch):
    from modules.sampling.context_windows import install
    p = _patcher(tiny_ltx)
    monkeypatch.setattr(type(p._patcher.model), "map_context_window_to_modalities", lambda *a: None, raising=False)
    note = install(p, ON, KEY)
    handler = p._patcher.model_options["context_handler"]
    assert (handler.context_length, handler.context_overlap, handler.dim) == (19, 5, 2)
    assert "145 frames" in note
    from comfy.patcher_extension import WrappersMP
    assert f"{KEY}.prepare" in p._patcher.wrappers[WrappersMP.PREPARE_SAMPLING]
    assert f"{KEY}.freenoise" in p._patcher.wrappers[WrappersMP.SAMPLER_SAMPLE]


def test_it_comes_off_again_and_off_cleans_a_stale_handler(tiny_ltx, monkeypatch):
    from modules.sampling.context_windows import install
    p = _patcher(tiny_ltx)
    monkeypatch.setattr(type(p._patcher.model), "map_context_window_to_modalities", lambda *a: None, raising=False)
    install(p, ON, KEY)
    assert patching.strip(p._patcher, KEY) >= 3
    assert "context_handler" not in p._patcher.model_options
    install(p, ON, KEY)
    install(p, {**ON, "enabled": False}, KEY)           # switched off on a model that carries the old handler
    assert "context_handler" not in p._patcher.model_options


def test_a_core_without_audio_video_windowing_is_refused_in_words(tiny_ltx, monkeypatch):
    from modules.sampling.context_windows import install
    p = _patcher(tiny_ltx)
    monkeypatch.delattr(type(p._patcher.model), "map_context_window_to_modalities", raising=False)
    with pytest.raises(RuntimeError, match="map_context_window_to_modalities"):
        install(p, ON, KEY)


def test_old_schedule_spellings_still_work_and_junk_is_named(tiny_ltx, monkeypatch):
    from modules.sampling.context_windows import install
    p = _patcher(tiny_ltx)
    monkeypatch.setattr(type(p._patcher.model), "map_context_window_to_modalities", lambda *a: None, raising=False)
    assert "standard_uniform" in install(p, {**ON, "schedule": "uniform_standard"}, KEY)
    with pytest.raises(RuntimeError, match="not one this ComfyUI knows"):
        install(p, {**ON, "schedule": "wobbly"}, KEY)


@pytest.mark.parametrize("length, overlap", [(33, 40), (145, 152), (9, 40)])
def test_an_overlap_as_long_as_the_window_is_cut_and_said(tiny_ltx, monkeypatch, length, overlap):
    from modules.sampling.context_windows import install
    p = _patcher(tiny_ltx)
    monkeypatch.setattr(type(p._patcher.model), "map_context_window_to_modalities", lambda *a: None, raising=False)
    note = install(p, {**ON, "length": length, "overlap": overlap}, KEY)
    handler = p._patcher.model_options["context_handler"]
    assert handler.context_overlap < handler.context_length and "overlap cut" in note
