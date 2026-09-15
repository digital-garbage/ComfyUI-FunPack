"""The settings card: what goes on it, and what deliberately does not.

The card exists to answer "which model was that?" months later, so the tests
that matter are about honesty -- a wired input must not be printed as if
someone typed it, and a value that is a list (a LoRA stack, say) must come
out as rows rather than as a wall of JSON.

Ported from v4's movie_editor/tests/test_settings_card.py, cut down the same
way core/settings_card.py itself was: no FunPackStudio/Chain-Sampler shape to
test here, since v5's pipeline has no such nodes -- every slot is walked the
same generic way, so there is one section kind, not several.
"""
import json

from core import settings_card as sc

HOST = {
    "python": "3.12.4",
    "comfyui": "0.30.1",
    "torch": {"version": "2.12.0+cu128", "cuda": "12.8", "attention": "SageAttention 3.0"},
    "gpus": [{"name": "RTX PRO 6000", "vram_gb": 95.0, "capability": "sm_120"}],
}


def _rows(report, title):
    return dict(next(s for s in report["sections"] if s["title"] == title)["rows"])


# --- what is on the card --------------------------------------------------------------


def test_full_filenames_are_not_abbreviated():
    """A truncated checkpoint name is the one thing that makes the card useless."""
    name = "MiniMax-H3-fl2va-bf16-Q6_K-a-really-long-name.gguf"
    rep = sc.collect(
        [{"id": "model", "node": "FunPackDiffusionModelLoader",
          "inputs": {"model_name": name, "attention": "sage3"}}], HOST)
    assert _rows(rep, "model")["model_name"] == name


def test_host_carries_torch_cuda_and_attention():
    rep = sc.collect([], HOST)
    host = dict(rep["host"])
    assert host["PyTorch"] == "2.12.0+cu128"
    assert host["CUDA"] == "12.8"
    assert host["Attention"] == "SageAttention 3.0"
    assert "sm_120" in host["GPU"] and "95.0 GB" in host["GPU"]


def test_a_cpu_only_box_says_so_rather_than_showing_nothing():
    """An absent GPU row reads as 'the probe failed'. Naming the absence does not."""
    rep = sc.collect([], {"torch": {}, "gpus": []})
    assert "none visible" in dict(rep["host"])["GPU"]


def test_lists_come_out_as_rows_and_settings():
    """A funpack_list widget's value is one JSON string. Printing it raw would
    technically be complete and practically unreadable."""
    rows = json.dumps([{"lora": "h3_turbo.safetensors", "strength": 1.0, "on": True},
                       {"lora": "style.safetensors", "strength": 0.65, "on": False}])
    rep = sc.collect(
        [{"id": "l", "node": "FunPackLoraLoader", "inputs": {"lora_list": rows}}], HOST)
    text = " ".join(v for _, v in rep["sections"][0]["rows"])
    assert "h3_turbo.safetensors" in text and "strength=1.0" in text
    assert "style.safetensors" in text and "strength=0.65" in text
    assert "on=off" in text          # a disabled row is still on the card, marked disabled


def test_an_empty_list_says_none_instead_of_vanishing():
    rep = sc.collect([{"id": "l", "node": "X", "inputs": {"lora_list": "[]"}}], HOST)
    assert any("(none)" in v for _, v in rep["sections"][0]["rows"])


def test_a_string_that_merely_looks_like_json_is_left_alone():
    """A prompt beginning with '[' must not be mangled into fake list rows."""
    rep = sc.collect([{"id": "n", "node": "X", "inputs": {"text": "[not json at all"}}], HOST)
    assert _rows(rep, "n")["text"] == "[not json at all"


def test_custom_nodes_appear_with_their_class():
    """The class is what someone reproducing this has to go and install."""
    rep = sc.collect([{"id": "n1", "node": "ImageUpscaleWithModel",
                       "inputs": {"factor": 2.0}}], HOST)
    assert rep["sections"][0]["node_class"] == "ImageUpscaleWithModel"
    assert _rows(rep, "n1")["factor"] == "2.0"


def test_a_group_prefixes_the_section_title():
    rep = sc.collect([{"id": "clip", "node": "X", "group": "Loaders", "inputs": {}}], HOST)
    assert rep["sections"][0]["title"] == "Loaders · clip"


# --- what must NOT be presented as a typed value --------------------------------------


def test_a_wired_input_is_marked_wired_not_its_stale_widget_value():
    """v5 encodes a wire directly in the input's own value -- [source_id, index] -- rather
    than a separate input_sources/wires lookup. Printing it as a typed value would be a
    confident lie: the widget behind a connected socket keeps its last typed value, and
    generation ignores it."""
    rep = sc.collect(
        [{"id": "up", "node": "X", "inputs": {}},
         {"id": "down", "node": "Y", "inputs": {"model": ["up", 0], "steps": 8}}], HOST)
    rows = _rows(rep, "down")
    assert "wired from up" in rows["model"]
    assert rows["steps"] == "8"


def test_control_after_generate_is_dropped():
    """It is a UI affordance, not a setting -- it says nothing about the render."""
    rep = sc.collect(
        [{"id": "a", "node": "X", "inputs": {"seed": 42, "control_after_generate": "randomize"}}],
        HOST)
    assert "control_after_generate" not in _rows(rep, "a")


def test_booleans_read_as_on_off_not_python():
    rep = sc.collect(
        [{"id": "a", "node": "X", "inputs": {"sla": True, "fp16_accumulation": False}}], HOST)
    rows = _rows(rep, "a")
    assert rows["sla"] == "on" and rows["fp16_accumulation"] == "off"


# --- bounding what an unbounded client can force this route to render ------------------


def test_a_very_long_value_is_truncated_not_rendered_whole():
    """extensive_testing found a single 200k-char value forced a 77,400px-tall
    image and a multi-second render -- this machine's own GPU box, reachable
    over a rental's plain-http IP with no auth, same as every other pipeline
    route. A card exists to be a compact summary; nobody reads a 200k-char
    row on it, so truncating is a real limit, not a corner cut short."""
    rep = sc.collect([{"id": "a", "node": "X", "inputs": {"text": "x" * 200_000}}], HOST)
    value = _rows(rep, "a")["text"]
    assert len(value) < 500
    assert "more chars" in value


def test_a_pipeline_with_an_unreasonable_slot_count_is_capped_not_rendered_whole():
    slots = [{"id": f"s{i}", "node": "X", "inputs": {"a": "v"}} for i in range(5000)]
    rep = sc.collect(slots, HOST)
    total_rows = sum(len(s["rows"]) for s in rep["sections"])
    assert total_rows <= sc._MAX_ROWS + 1   # +1 for the truncation note's own row
    assert "omitted" in rep["sections"][-1]["rows"][0][1]


def test_a_funpack_list_with_an_unreasonable_row_count_is_capped():
    import json
    rows = json.dumps([{"i": i} for i in range(5000)])
    rep = sc.collect([{"id": "l", "node": "X", "inputs": {"lora_list": rows}}], HOST)
    text = " ".join(v for _, v in rep["sections"][0]["rows"])
    assert "more rows omitted" in text


def test_many_funpack_lists_on_one_slot_still_respect_the_card_wide_cap():
    """extensive_testing round 2: the per-card check used to fire once per
    SLOT, so one slot with many funpack_list inputs -- each individually
    capped at _MAX_ROWS by _rows_from_list -- blew the card-wide budget by a
    large multiple before the next slot's check ever ran. 50 inputs x 2000
    rows each used to produce 100,050 total rows from a single slot."""
    import json
    big_list = json.dumps([{"i": i} for i in range(2000)])
    inputs = {f"list_{n}": big_list for n in range(50)}
    rep = sc.collect([{"id": "s", "node": "X", "inputs": inputs}], HOST)
    total_rows = sum(len(s["rows"]) for s in rep["sections"])
    assert total_rows <= sc._MAX_ROWS + 5
    assert "truncated" in rep["sections"][-1]["title"]


def test_render_png_stays_fast_even_when_both_caps_are_individually_honored():
    """extensive_testing round 2: a request that honors BOTH _MAX_ROWS and
    _MAX_VALUE_CHARS exactly -- 2000 rows of 400-char, space-free values --
    still cost ~14s and ~10MB, because nothing bounded their product. This is
    the test the round-1 fix should have had: it actually calls render_png,
    not just collect."""
    import time
    slots = [{"id": f"s{i}", "node": "X", "inputs": {"a": "x" * sc._MAX_VALUE_CHARS}}
              for i in range(2000)]
    rep = sc.collect(slots, HOST)
    start = time.monotonic()
    png = sc.render_png(rep, "dark")
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
    assert len(png) < 5_000_000


def test_render_png_stays_fast_when_one_slot_has_many_large_lists():
    """The Finding B composition gap, carried all the way through to the
    actual render (67.8s / 123MB before the fix)."""
    import json
    import time
    big_list = json.dumps([{"i": i} for i in range(2000)])
    inputs = {f"list_{n}": big_list for n in range(50)}
    rep = sc.collect([{"id": "s", "node": "X", "inputs": inputs}], HOST)
    start = time.monotonic()
    png = sc.render_png(rep, "dark")
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
    assert len(png) < 5_000_000


# --- rendering --------------------------------------------------------------------------


def test_render_png_produces_a_real_png_with_the_report_embedded():
    rep = sc.collect(
        [{"id": "model", "node": "FunPackDiffusionModelLoader",
          "inputs": {"model_name": "h3.safetensors"}}], HOST, project_name="My project")
    png = sc.render_png(rep, "dark")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    from PIL import Image
    img = Image.open(__import__("io").BytesIO(png))
    assert img.format == "PNG"
    embedded = json.loads(img.text["funpack_settings"])
    assert embedded["project"] == "My project"
    assert embedded["sections"][0]["rows"] == [("model_name", "h3.safetensors")] or \
        embedded["sections"][0]["rows"] == [["model_name", "h3.safetensors"]]


def test_light_theme_is_a_real_choice_not_a_silent_fallback():
    rep = sc.collect([], HOST)
    dark = sc.render_png(rep, "dark")
    light = sc.render_png(rep, "light")
    assert dark != light


def test_an_unknown_theme_falls_back_to_dark_rather_than_crashing():
    rep = sc.collect([], HOST)
    sc.render_png(rep, "not-a-real-theme")  # must not raise
