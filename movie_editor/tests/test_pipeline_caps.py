"""Tests for pipeline capability flags (Studio / Chain Sampler availability)."""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend import pipeline_caps  # noqa: E402
from movie_editor.backend.timeline import Project, Scene, SceneSource  # noqa: E402


def _scene(stype: str, ref: str | None = None) -> Scene:
    return Scene(
        id="s1",
        text="x",
        source=SceneSource(type=stype, media_ref=ref),
    )


def test_disable_core_turns_off_studio_and_chain():
    p = Project(name="t")
    m = {"disable_core": True, "workflow_import": {"name": "W"}}
    c = pipeline_caps.capabilities(p, m)
    assert c["studio"] is False
    assert c["chain_sampler"] is False


def test_custom_slots_without_disable_core():
    p = Project(name="t", conditioning_slot="custom1", sampler_slot="custom2")
    c = pipeline_caps.capabilities(p, {"slots": []})
    assert c["studio"] is False
    assert c["chain_sampler"] is False


def test_effective_source_fallback_without_chain():
    sc = _scene("carry")
    assert pipeline_caps.effective_source_type(sc, False) == "empty"
    sc2 = _scene("mixed", "img1")
    assert pipeline_caps.effective_source_type(sc2, False) == "empty"
    sc3 = _scene("image", "img1")
    assert pipeline_caps.effective_source_type(sc3, False) == "image"


def test_effective_source_unchanged_with_chain():
    sc = _scene("carry")
    assert pipeline_caps.effective_source_type(sc, True) == "carry"


def test_source_needs_anchor_media_by_mode():
    assert pipeline_caps.source_needs_anchor_media(_scene("image"), True) is True
    assert pipeline_caps.source_needs_anchor_media(_scene("mixed"), True) is True
    assert pipeline_caps.source_needs_anchor_media(_scene("anchor_guide"), True) is True
    assert pipeline_caps.source_needs_anchor_media(_scene("carry"), True) is False
    assert pipeline_caps.source_needs_anchor_media(_scene("empty"), True) is False


def test_scenes_missing_anchor_media_flags_only_the_unset_ones():
    """A ref that IS set but has left the media bin is server-side
    _missing_scene_anchor_media's job; this catches the ref never being set."""
    p = Project(name="t")
    ok = _scene("image", "img1"); ok.id = "s_ok"
    bad = _scene("image"); bad.id = "s_bad"
    carry = _scene("carry"); carry.id = "s_carry"
    skipped = _scene("mixed"); skipped.id = "s_excluded"; skipped.excluded = True
    p.scenes = [ok, bad, carry, skipped]
    assert pipeline_caps.scenes_missing_anchor_media(p, True) == ["s_bad"]


def test_scenes_missing_anchor_media_silent_in_t2v():
    """A t2v project starts shots from the prompt, so an unset anchor is the norm."""
    p = Project(name="t", generation_mode="t2v")
    bad = _scene("image"); bad.id = "s_bad"
    p.scenes = [bad]
    assert pipeline_caps.is_t2v(p) is True
    # from_dict is an explicit field list — a field missing from it round-trips to the default.
    assert Project.from_dict(p.to_dict()).generation_mode == "t2v"
    assert Project.from_dict({"name": "x"}).generation_mode == "i2v"
    assert pipeline_caps.scenes_missing_anchor_media(p, True) == []
    # ...and the default is unchanged for everyone else.
    assert pipeline_caps.is_t2v(Project(name="t")) is False


def test_scenes_missing_anchor_media_silent_without_chain_sampler():
    """Without Chain Sampler an anchorless image scene degrades to t2v by design,
    so warning about a missing anchor there would be noise."""
    p = Project(name="t")
    bad = _scene("image"); bad.id = "s_bad"
    p.scenes = [bad]
    assert pipeline_caps.scenes_missing_anchor_media(p, False) == []


# ── MiniMax H3 family ─────────────────────────────────────────────────────────

def test_family_is_explicit_and_falls_back_rather_than_guessing():
    from movie_editor.backend import pipeline_wiring as pw
    assert pw.family_of({"model_family": "minimax_h3"}) == "minimax_h3"
    assert pw.family_of({"model_family": "MiniMax_H3"}) == "minimax_h3"
    assert pw.family_of({"model_family": "hailuo-9000"}) == "ltxav"
    assert pw.family_of({}) == "ltxav"
    assert pw.family_of(None) == "ltxav"


def test_h3_moves_the_audio_and_latent_ports_onto_different_nodes():
    from movie_editor.backend import pipeline_wiring as pw
    ltx = pw.allowed_port_ids("audio_vae", "VAE", family="ltxav")
    h3 = pw.allowed_port_ids("audio_vae", "VAE", family="minimax_h3")
    assert ltx == ["LTXVAudioVAEDecode.audio_vae"]
    # H3 decodes with core's generic node, and the same VAE may also encode audio references
    assert h3 == ["VAEDecodeAudio.vae", "FunPackLTXAVSceneChainSampler.audio_vae"]

    # the empty latent no longer goes through Studio into Concat
    assert pw.allowed_port_ids("empty_latent", "LATENT", family="ltxav") == ["FunPackStudio.latent"]
    assert pw.allowed_port_ids("empty_latent", "LATENT", family="minimax_h3") == \
        ["FunPackLTXAVSceneChainSampler.latent_template"]

    # ... and there is no separate audio-encoder step at all
    assert pw.allowed_port_ids("audio_encoder", "LATENT", family="minimax_h3") == \
        ["FunPackLTXAVSceneChainSampler.latent_template"]  # falls through to the chain terminal


def test_h3_requirements_drop_the_audio_latent_and_require_the_av_latent():
    from movie_editor.backend import nodes
    ltx = {r["id"]: r for r in nodes.pipeline_requirements("ltxav")}
    h3 = {r["id"]: r for r in nodes.pipeline_requirements("minimax_h3")}
    assert ltx["audio_latent"]["required"] is True
    assert "audio_latent" not in h3                      # one node makes both streams
    assert ltx["init_latent"]["required"] is False
    assert h3["init_latent"]["required"] is True         # nothing else produces one
    assert "MiniMax H3" in h3["init_latent"]["hint"]
    # the shared requirements keep their ids so existing UI keys still resolve
    assert set(h3) <= set(ltx)


def test_h3_does_not_offer_ports_on_nodes_its_graph_never_emits():
    from movie_editor.backend import nodes
    oi = {
        "LTXVConditioning": {"input": {"required": {"positive": ["CONDITIONING"]}}, "output": []},
        "LTXVConcatAVLatent": {"input": {"required": {"audio_latent": ["LATENT"]}}, "output": []},
        "LTXVAudioVAEDecode": {"input": {"required": {"audio_vae": ["VAE"]}}, "output": []},
        "VAEDecodeAudio": {"input": {"required": {"vae": ["VAE"]}}, "output": []},
    }
    h3_ports = {p["id"] for p in nodes.pipeline_ports(oi, "minimax_h3")}
    assert not any(p.startswith("LTXVConcatAVLatent.") for p in h3_ports)
    assert not any(p.startswith("LTXVAudioVAEDecode.") for p in h3_ports)
    assert "VAEDecodeAudio.vae" in h3_ports

    ltx_ports = {p["id"] for p in nodes.pipeline_ports(oi, "ltxav")}
    assert "LTXVAudioVAEDecode.audio_vae" in ltx_ports
    assert "VAEDecodeAudio.vae" not in ltx_ports


# ── Simple mode ───────────────────────────────────────────────────────────────

def test_simple_mode_switches_refinement_off():
    si, ss = pipeline_caps.apply_simple_mode(
        {"embed_guidance": True, "dynashift": True, "score_slider": True,
         "taste_nearest_prompt": True, "output_guidance": True, "cfg": 1.0},
        {"refiner": {"value_guidance": True, "steer_mode": "absolute", "vision_conditioning": True}},
    )
    for key in ("embed_guidance", "dynashift", "score_slider", "taste_nearest_prompt",
                "output_guidance"):
        assert si[key] is False, key
    assert ss["refiner"]["value_guidance"] is False
    assert ss["refiner"]["steer_mode"] == "relative"
    # untouched: not refinement, just how the scene is generated
    assert si["cfg"] == 1.0
    assert ss["refiner"]["vision_conditioning"] is True


def test_simple_mode_leaves_everything_that_is_not_refinement_alone():
    """Cross-shot memory, guides, experimental sampling and the second pass all work here.
    None of them need a rating to do their job, so none of them are Simple mode's business —
    and the second pass in particular was allowed by Easy Gen, which this mode replaced."""
    on = {
        "second_pass": True, "second_pass_op": "upscale_2x",
        "mid_scene_guide": True, "joyai_memory": True, "joyai_audio_memory": True,
        "carry_i2v_guides": True, "alg_anchor": True, "alg_blur_guides": True,
        "bounded_attention_enabled": True, "identity_transfer_enabled": True,
        "segmented_detailing": True, "plateau_cache": True, "context_windows": True,
    }
    si, _ss = pipeline_caps.apply_simple_mode(dict(on), {})
    for key, value in on.items():
        assert si[key] == value, f"{key} was stripped but is not refinement"


def test_simple_mode_stops_the_velocity_bank_replaying():
    """velocity_bias / rescue live per-sampler inside studio_settings and are hidden in
    Simple mode. Hidden AND live is the worst of both."""
    ss_in = {"samplers": {"high": {"velocity_bias_mode": "apply", "rescue_mode": True, "eta": 1.0},
                          "low": {"velocity_bias_mode": "capture_and_apply", "rescue_mode": True}}}
    _si, ss = pipeline_caps.apply_simple_mode({}, ss_in)
    for entry in ss["samplers"].values():
        assert entry["velocity_bias_mode"] == "off"
        assert entry["rescue_mode"] is False
    assert ss["samplers"]["high"]["eta"] == 1.0     # nothing else touched


def test_the_strip_list_matches_what_the_ui_hides():
    """engine_settings.js hides exactly the rating-gated controls in Simple mode. A key
    stripped here but visible there is a control that lies; visible there but live here is
    the same lie inverted. This pins the backend half against the frontend source."""
    js = (ROOT / "movie_editor" / "frontend" / "engine_settings.js").read_text(encoding="utf-8")
    hidden = set(re.findall(r'"([a-z0-9_]+)"', js[js.index("RATING_GATED_KNOBS"):js.index("let _mounted")]))
    for key in pipeline_caps.SIMPLE_MODE_SAMPLER_OFF:
        assert key in hidden, f"{key} is stripped in Simple mode but its control is not hidden"
    for key in pipeline_caps.SIMPLE_MODE_REFINER_OFF:
        assert key in hidden, f"{key} is stripped in Simple mode but its control is not hidden"


def test_simple_mode_does_not_mutate_the_project_settings():
    """The project keeps what the Editor set — only the run is stripped."""
    stored_si = {"embed_guidance": True}
    stored_ss = {"refiner": {"value_guidance": True}}
    pipeline_caps.apply_simple_mode(stored_si, stored_ss)
    assert stored_si == {"embed_guidance": True}
    assert stored_ss == {"refiner": {"value_guidance": True}}


def test_simple_mode_handles_an_empty_project():
    si, ss = pipeline_caps.apply_simple_mode(None, None)
    assert si["embed_guidance"] is False
    assert ss["refiner"]["value_guidance"] is False
