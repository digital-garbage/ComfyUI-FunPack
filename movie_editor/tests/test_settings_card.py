"""The settings card: what goes on it, and what deliberately does not.

The card exists to answer "which model was that?" months later, so the tests that matter are
about honesty — a wired input must not be printed as if someone typed it, and a value that
is a list of LoRAs must come out as LoRAs rather than as a wall of JSON.
"""
import json

import pytest

from movie_editor.backend import settings_card as sc


HOST = {
    "python": "3.12.4",
    "comfyui": "0.30.1",
    "torch": {"version": "2.12.0+cu128", "cuda": "12.8", "attention": "SageAttention 3.0"},
    "gpus": [{"name": "RTX PRO 6000", "vram_gb": 95.0, "capability": "sm_120"}],
}


def _models(*slots):
    return {"model_family": "minimax_h3", "slots": list(slots)}


def _rows(report, title):
    return dict(next(s for s in report["sections"] if s["title"] == title)["rows"])


# --- what is on the card --------------------------------------------------------------


def test_full_filenames_are_not_abbreviated():
    """A truncated checkpoint name is the one thing that makes the card useless."""
    name = "MiniMax-H3-fl2va-bf16-Q6_K-a-really-long-name.gguf"
    rep = sc.collect(_models({"id": "a", "role_label": "Diffusion model",
                              "node_class": "FunPackDiffusionModelLoader",
                              "inputs": {"model_name": name, "attention": "sage3"}}), HOST)
    assert _rows(rep, "Diffusion model")["model_name"] == name


def test_host_carries_torch_cuda_and_attention():
    rep = sc.collect(_models(), HOST)
    host = dict(rep["host"])
    assert host["PyTorch"] == "2.12.0+cu128"
    assert host["CUDA"] == "12.8"
    assert host["Attention"] == "SageAttention 3.0"
    assert "sm_120" in host["GPU"] and "95.0 GB" in host["GPU"]


def test_a_cpu_only_box_says_so_rather_than_showing_nothing():
    """An absent GPU row reads as 'the probe failed'. Naming the absence does not."""
    rep = sc.collect(_models(), {"torch": {}, "gpus": []})
    assert "none visible" in dict(rep["host"])["GPU"]


def test_loras_come_out_as_files_and_weights():
    """The LoRA list is one JSON string widget. Printing it raw would technically be
    complete and practically unreadable."""
    rows = json.dumps([{"lora": "h3_turbo.safetensors", "type": "model", "strength": 1.0,
                        "on": True},
                       {"lora": "style.safetensors", "type": "both", "strength": 0.65,
                        "on": False}])
    rep = sc.collect(_models({"id": "l", "role_label": "LoRA", "node_class": "FunPackLoraLoader",
                              "inputs": {"lora_list": rows}}), HOST)
    text = " ".join(v for _, v in next(s for s in rep["sections"])["rows"])
    assert "h3_turbo.safetensors" in text and "strength=1.0" in text
    assert "style.safetensors" in text and "strength=0.65" in text
    assert "on=off" in text          # a disabled row is still on the card, marked disabled


def test_an_empty_list_says_none_instead_of_vanishing():
    rep = sc.collect(_models({"id": "l", "role_label": "LoRA", "node_class": "X",
                              "inputs": {"lora_list": "[]"}}), HOST)
    assert any("(none)" in v for _, v in rep["sections"][0]["rows"])


def test_a_string_that_merely_looks_like_json_is_left_alone():
    """A prompt beginning with '[' must not be mangled into fake list rows."""
    rep = sc.collect(_models({"id": "n", "role_label": "Note", "node_class": "X",
                              "inputs": {"text": "[not json at all"}}), HOST)
    assert _rows(rep, "Note")["text"] == "[not json at all"


def test_custom_nodes_appear_with_their_class():
    """The class is what someone reproducing this has to go and install."""
    rep = sc.collect(_models({"id": "n1", "label": "Upscale",
                              "node_class": "ImageUpscaleWithModel",
                              "inputs": {"factor": 2.0}}), HOST)
    assert rep["sections"][0]["node_class"] == "ImageUpscaleWithModel"
    assert _rows(rep, "Upscale")["factor"] == "2.0"


# --- what must NOT be presented as a typed value --------------------------------------


def test_an_input_with_a_named_source_is_marked_wired():
    """The widget behind a connected socket keeps its last typed value, and generation
    ignores it. Printing it as the setting would be a confident lie."""
    rep = sc.collect(_models({"id": "a", "role_label": "Sampler", "node_class": "X",
                              "inputs": {"model": "stale.safetensors", "steps": 8},
                              "input_sources": {"model": "out:fp_unet:MODEL"}}), HOST)
    rows = _rows(rep, "Sampler")
    assert rows["model"] == "‹wired›"
    assert rows["steps"] == "8"


def test_an_input_fed_by_another_slots_wire_is_marked_wired():
    """The other direction: the producer names the target, and the consumer says nothing."""
    rep = sc.collect(_models(
        {"id": "up", "role_label": "Loader", "node_class": "X", "inputs": {},
         "wires": {"MODEL": ["node:down:model"]}},
        {"id": "down", "role_label": "Sampler", "node_class": "Y",
         "inputs": {"model": "stale.safetensors"}}), HOST)
    assert _rows(rep, "Sampler")["model"] == "‹wired›"


def test_auto_is_not_a_source():
    """'auto' means no choice was made, so the widget IS the value."""
    rep = sc.collect(_models({"id": "a", "role_label": "Node", "node_class": "X",
                              "inputs": {"vae": "ae.safetensors"},
                              "input_sources": {"vae": "auto"}}), HOST)
    assert _rows(rep, "Node")["vae"] == "ae.safetensors"


def test_control_after_generate_is_dropped():
    """It is a UI affordance, not a setting — it says nothing about the render."""
    rep = sc.collect(_models({"id": "a", "role_label": "Node", "node_class": "X",
                              "inputs": {"seed": 42, "control_after_generate": "randomize"}}),
                     HOST)
    assert "control_after_generate" not in _rows(rep, "Node")


def test_booleans_read_as_on_off_not_python():
    rep = sc.collect(_models({"id": "a", "role_label": "Node", "node_class": "X",
                              "inputs": {"sla": True, "fp16_accumulation": False}}), HOST)
    rows = _rows(rep, "Node")
    assert rows["sla"] == "on" and rows["fp16_accumulation"] == "off"


# --- rendering ------------------------------------------------------------------------


@pytest.fixture
def report():
    return sc.collect(
        _models({"id": "a", "role_label": "Diffusion model",
                 "node_class": "FunPackDiffusionModelLoader",
                 "inputs": {"model_name": "x.safetensors", "attention": "sage3"}}),
        HOST, project_name="Night Drive", version="4.0.0", codename="Blinding Blackout")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_renders_a_png_in_each_theme(report, theme):
    png = sc.render_png(report, theme)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_two_themes_are_actually_different(report):
    """A theme parameter that renders the same picture is a parameter that does nothing."""
    assert sc.render_png(report, "dark") != sc.render_png(report, "light")


def test_an_unknown_theme_falls_back_rather_than_failing(report):
    assert sc.render_png(report, "chartreuse")[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_card_grows_with_its_content(report):
    """Height is laid out from the content, so a long pipeline must not be cropped and a
    short one must not sit above a field of empty space."""
    from PIL import Image
    import io

    big = sc.collect(_models(*[
        {"id": f"n{i}", "role_label": f"Node {i}", "node_class": "X", "inputs": {"v": i}}
        for i in range(20)]), HOST)
    short = Image.open(io.BytesIO(sc.render_png(report, "dark"))).height
    tall = Image.open(io.BytesIO(sc.render_png(big, "dark"))).height
    assert tall > short


def test_the_report_is_embedded_for_machines(report):
    """The picture is also the record: the JSON rides in a tEXt chunk."""
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(sc.render_png(report, "dark")))
    payload = json.loads(img.text["funpack_settings"])
    assert payload["sections"][0]["node_class"] == "FunPackDiffusionModelLoader"


def test_the_watermark_names_the_version(report):
    """It is what tells you whether the card predates a behaviour change."""
    from PIL import Image
    import io

    # Rendered text is not readable back out of pixels; the embedded record is the
    # checkable half, and the footer is drawn from these same two fields.
    img = Image.open(io.BytesIO(sc.render_png(report, "dark")))
    payload = json.loads(img.text["funpack_settings"])
    assert payload["version"] == "4.0.0" and payload["codename"] == "Blinding Blackout"


def test_long_values_wrap_instead_of_running_off_the_card():
    lines = sc._wrap("a" * 200, 40)
    assert all(len(x) <= 40 for x in lines)
    assert "".join(lines) == "a" * 200


# ── sampling settings ─────────────────────────────────────────────────────────
# Two runs on the same checkpoint with different schedules are different renders, so the
# model files alone do not answer "what produced this".


STUDIO = {"studio_settings": json.dumps({
    "samplers": {
        "high": {"type": "Hybrid Euler 2S", "sigmas": "1.0, 0.5, 0.0",
                 "hybrid": {"eta": 0.7, "quality_sharpness": 0.3},
                 "distilled": {"order": 2}, "normalizing": {"normalize_strength": 0.5}},
        "low": {"type": "KSampler", "sigmas": "0.4, 0.0", "ksampler_name": "res_multistep"},
    },
    "refiner": {"mode": "Refine", "negative_erase": True, "negative_erase_strength": 0.5},
})}


def _card(**kw):
    return sc.collect(_models(), HOST, **kw)


def _titles(rep):
    return [s["title"] for s in rep["sections"]]


def test_the_sampler_and_its_schedule_are_on_the_card():
    rows = dict(next(s for s in _card(studio_inputs=STUDIO)["sections"]
                     if s["title"] == "Sampler")["rows"])
    assert rows["algorithm"] == "Hybrid Euler 2S"
    assert rows["sigmas"] == "1.0, 0.5, 0.0"


def test_only_the_selected_algorithms_settings_are_shown():
    """A pass config always carries hybrid/distilled/normalizing blocks; only one is live.
    Printing all three is the same lie as printing a wired input's stale widget."""
    rows = dict(next(s for s in _card(studio_inputs=STUDIO)["sections"]
                     if s["title"] == "Sampler")["rows"])
    assert rows["    eta"] == "0.7"
    assert not any("order" in k for k in rows)              # Distilled Flow is not selected
    assert not any("normalize_strength" in k for k in rows)


def test_a_computed_schedule_shows_steps_instead_of_sigmas():
    studio = {"studio_settings": json.dumps({"samplers": {"high": {
        "type": "KSampler", "scheduler": "karras", "steps": 12, "ksampler_name": "euler"}}})}
    rows = dict(next(s for s in _card(studio_inputs=studio)["sections"]
                     if s["title"] == "Sampler")["rows"])
    assert rows["schedule"] == "karras" and rows["steps"] == "12"
    assert "sigmas" not in rows


def test_ksampler_sharpness_is_only_listed_when_it_is_on():
    on = {"studio_settings": json.dumps({"samplers": {"high": {
        "type": "KSampler", "ksampler_sharpness": 0.3}}})}
    off = {"studio_settings": json.dumps({"samplers": {"high": {"type": "KSampler"}}})}
    assert any("quality_sharpness" in k for k, _ in
               next(s for s in _card(studio_inputs=on)["sections"]
                    if s["title"] == "Sampler")["rows"])
    assert not any("quality_sharpness" in k for k, _ in
                   next(s for s in _card(studio_inputs=off)["sections"]
                        if s["title"] == "Sampler")["rows"])


def test_the_second_pass_appears_only_when_it_is_running():
    """An off feature's settings on the card read as if they were in effect."""
    assert "Second pass" not in _titles(_card(studio_inputs=STUDIO))
    on = _card(studio_inputs=STUDIO, sampler_inputs={"second_pass": True})
    assert "Second pass" in _titles(on)
    rows = dict(next(s for s in on["sections"] if s["title"] == "Second pass")["rows"])
    assert rows["own sampler"].startswith("no")             # own_sampler not set
    assert rows["sigmas"] == "0.4, 0.0"


def test_a_second_pass_with_its_own_sampler_says_which():
    studio = {"studio_settings": json.dumps({"samplers": {
        "high": {"type": "Hybrid Euler 2S"},
        "low": {"type": "KSampler", "own_sampler": True, "ksampler_name": "res_multistep"}}})}
    rows = dict(next(s for s in _card(studio_inputs=studio,
                                      sampler_inputs={"second_pass": True})["sections"]
                     if s["title"] == "Second pass")["rows"])
    assert rows["own sampler"] == "yes"
    assert rows["sampler_name"] == "res_multistep"


def test_studio_and_chain_sampler_overrides_are_listed():
    rep = _card(studio_inputs=STUDIO, sampler_inputs={"alg_anchor": True, "second_pass": False})
    assert dict(next(s for s in rep["sections"]
                     if s["title"] == "Studio")["rows"])["negative_erase"] == "on"
    assert dict(next(s for s in rep["sections"]
                     if s["title"] == "Chain Sampler")["rows"])["alg_anchor"] == "on"


def test_the_render_geometry_leads_the_card():
    rep = _card(render={"size": "768x512", "frame rate": 25})
    assert rep["sections"][0]["title"] == "Render"
    assert dict(rep["sections"][0]["rows"])["size"] == "768x512"


def test_a_project_with_no_sampling_settings_adds_no_empty_sections():
    assert _titles(_card()) == _titles(sc.collect(_models(), HOST))


def test_unparseable_studio_settings_do_not_break_the_card():
    rep = _card(studio_inputs={"studio_settings": "{not json"})
    assert sc.render_png(rep, "dark")[:8] == b"\x89PNG\r\n\x1a\n"
