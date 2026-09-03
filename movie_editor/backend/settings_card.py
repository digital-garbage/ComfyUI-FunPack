"""Export the loaded pipeline as a PNG you can post next to a render.

"Which model was that?" is the most common question asked about a good generation and the
hardest to answer three days later. This renders the answer as a picture: every loader with
its full filename, every LoRA with its weight, the typed-in values of any custom node, and
the host facts (torch, CUDA, attention) that decide whether the setup is reproducible at all
on another box.

Two decisions worth knowing:

* **Only inputs that were TYPED are listed.** A wired input's value is whatever the upstream
  node produced, so printing the stale widget still sitting behind the socket would be a
  confident lie. Wired inputs are named as wired instead of being dropped, because "this
  node has a VAE input" is part of the answer.
* **The JSON is embedded in the PNG's tEXt chunk** under `funpack_settings`, so the picture
  is also machine-readable. Nothing reads it back yet; it costs a few hundred bytes and it
  is the difference between a screenshot and a record.

Rendering is best-effort about fonts only. Everything else is derived from the project's own
models config, so this module never has to guess.
"""
from __future__ import annotations

import io
import json
from datetime import datetime

# Straight from styles.css, so the card matches the app it was exported from rather than
# approximating it. Only the tokens a flat document needs.
THEMES = {
    "dark": {
        "bg": "#0c0b09", "panel": "#16140f", "text": "#ece7db", "muted": "#918a7a",
        "faint": "#645e51", "line": "#2a261e", "accent": "#f3a93c",
    },
    "light": {
        "bg": "#f5f7fa", "panel": "#ffffff", "text": "#16202c", "muted": "#59677a",
        "faint": "#8d9aab", "line": "#d3dae3", "accent": "#2f7fd4",
    },
}

# Inputs every node carries that say nothing about the setup.
_NOISE_INPUTS = {"control_after_generate"}


def _is_wired(slot: dict, name: str, slots: list) -> bool:
    """Whether `name` is fed by something rather than typed.

    Two ways an input can be fed: the slot names its own source (`input_sources`), or another
    slot wires an output at it. "auto" is not a source — it is the absence of a choice.
    """
    src = (slot.get("input_sources") or {}).get(name)
    if src and src != "auto":
        return True
    target = f"node:{slot.get('id')}:{name}"
    for other in slots or []:
        if other is slot or other.get("id") == slot.get("id"):
            continue
        for raw in (other.get("wires") or {}).values():
            targets = raw if isinstance(raw, list) else ([raw] if raw else [])
            if target in targets:
                return True
    return False


def _rows_from_list(value):
    """A funpack_list widget's JSON array as [(label, value)], or None if it isn't one.

    Detected by content rather than by asking object_info: the value has to be parsed to be
    printed anyway, and a card that needs a live ComfyUI to render is a card you cannot
    export from a broken install.
    """
    if not isinstance(value, str) or not value.strip().startswith("["):
        return None
    try:
        rows = json.loads(value)
    except Exception:
        return None
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return None
    out = []
    for i, row in enumerate(rows, 1):
        # The first string field is the identity of the row (the LoRA file, the encoder);
        # everything else is its settings, printed after it in declaration order.
        parts = [f"{k}={_short(v)}" for k, v in row.items()]
        out.append((f"[{i}]", "  ".join(parts) if parts else "—"))
    return out          # [] for an empty list; the caller collapses that onto one line


def _short(value):
    if isinstance(value, bool):
        return "on" if value else "off"
    if value is None or value == "":
        return "—"
    return str(value)


def _host_rows(host: dict) -> list:
    """The facts that decide whether this setup reproduces elsewhere."""
    host = host or {}
    torch = host.get("torch") or {}
    gpus = host.get("gpus") or []
    rows = [
        ("PyTorch", _short(torch.get("version"))),
        ("CUDA", _short(torch.get("cuda"))),
        ("Attention", _short(torch.get("attention"))),
        ("Python", _short(host.get("python"))),
        ("ComfyUI", _short(host.get("comfyui"))),
    ]
    for gpu in gpus:
        label = _short(gpu.get("name"))
        extra = [x for x in (gpu.get("capability"),
                             f"{gpu['vram_gb']} GB" if gpu.get("vram_gb") else None) if x]
        rows.append(("GPU", f"{label}  ({', '.join(extra)})" if extra else label))
    if not gpus:
        rows.append(("GPU", "none visible (CPU / MPS)"))
    return rows


# The algorithm sub-blocks a pass config always carries. Only the one matching `type` is
# live; printing the other two would be the same lie as printing a wired input's stale widget.
_PASS_BLOCKS = {"Hybrid Euler 2S": "hybrid", "Distilled Flow": "distilled",
                "Normalizing": "normalizing"}

# Widget names that say nothing about the render, or that are stored elsewhere on the card.
_SKIP_STUDIO_KEYS = {"studio_settings"}


def _pass_rows(cfg: dict) -> list:
    """One sampling pass: its schedule, then only the settings its own algorithm reads."""
    cfg = cfg or {}
    kind = str(cfg.get("type") or "Hybrid Euler 2S")
    rows = [("algorithm", kind)]
    scheduler = cfg.get("scheduler", cfg.get("ksampler_scheduler"))
    if scheduler and scheduler != "use_user_sigmas":
        rows.append(("schedule", _short(scheduler)))
        rows.append(("steps", _short(cfg.get("steps", cfg.get("ksampler_steps")))))
    else:
        rows.append(("sigmas", _short(cfg.get("sigmas"))))
    if kind == "KSampler":
        rows.append(("sampler_name", _short(cfg.get("ksampler_name", "euler"))))
        if float(cfg.get("ksampler_sharpness") or 0) > 0:
            rows.append(("quality_sharpness", _short(cfg.get("ksampler_sharpness"))))
            rows.append(("sharpen_last_pct", _short(cfg.get("ksampler_sharpen_start_pct", 0.35))))
        return rows
    block = cfg.get(_PASS_BLOCKS.get(kind, "")) or {}
    for k, v in block.items():
        rows.append((f"    {k}", _short(v)))
    return rows


# ── what is worth printing ───────────────────────────────────────────────────
#
# The card answers "what produced this video". A setting left at its default produced
# nothing, and a setting belonging to a switched-off feature produced nothing either — a
# region-sharpening prompt printed under a disabled sharpener reads as if it ran. Both are
# noise, and enough of it buries the handful of values that DID decide the render.
#
# Two rules, both the user's: skip a value that equals the widget's own default, and skip a
# value whose feature is off.

#: knob -> (the toggle that owns it, the values of that toggle which switch it on).
#: None means "any truthy value". Mirrored from the Editor's own dependsOn wiring, and a
#: test regenerates this from engine_settings.js so the two cannot drift apart.
#:
#: A tuple, not one value: `absolute_strength` belongs to `steer_mode` and is live for BOTH
#: "absolute" and "both". Reading only the singular `dependsValue` missed the plural
#: `dependsVals` the Studio rows use, so it printed under `relative`, where it does nothing.
#: knobs the Editor renders by hand rather than from its knob table, so they carry no
#: dependsOn for the test below to find. Kept separate so the generated half stays generated.
_OWNED_EXTRA = {
    "identity_projector": ("identity_transfer_enabled", None),
    # Segmented detailing is no longer offered in the Editor (see detailing.py), but a raw
    # ComfyUI graph can still tick it, so the card still has to hide its settings when it
    # is off.
    "detail_targets": ("segmented_detailing", None),
    "detail_strength": ("segmented_detailing", None),
    "detail_threshold": ("segmented_detailing", None),
    "detail_max_area": ("segmented_detailing", None),
    "detail_mode": ("segmented_detailing", None),
}

_OWNED_BY = {
    **_OWNED_EXTRA,
    "absolute_strength": ("steer_mode", ('absolute', 'both')),
    "alg_anchor_sigma_threshold": ("alg_anchor", None),
    "alg_anchor_strength": ("alg_anchor", None),
    "alg_guide_blur_sigma_threshold": ("alg_blur_guides", None),
    "alg_guide_blur_strength": ("alg_blur_guides", None),
    "arcface_mode": ("identity_transfer_enabled", None),
    "context_window_freenoise": ("context_windows", None),
    "context_window_fuse": ("context_windows", None),
    "context_window_length": ("context_windows", None),
    "context_window_overlap": ("context_windows", None),
    "context_window_retain_first": ("context_windows", None),
    "context_window_schedule": ("context_windows", None),
    "debug_log": ("identity_transfer_enabled", None),
    "dynashift_strength": ("dynashift", None),
    "dynashift_threshold": ("dynashift", None),
    "embed_guidance_source": ("embed_guidance", None),
    "embed_guidance_strength": ("embed_guidance", None),
    "id_strength": ("identity_transfer_enabled", None),
    "joyai_audio_memory": ("joyai_memory", None),
    "joyai_fix_frames": ("joyai_memory", None),
    "joyai_frame_select": ("joyai_memory", None),
    "joyai_memory_size": ("joyai_memory", None),
    "joyai_memory_strength": ("joyai_memory", None),
    "negative_erase_mode": ("negative_erase", ('true',)),
    "negative_erase_renorm": ("negative_erase", ('true',)),
    "negative_erase_strength": ("negative_erase", ('true',)),
    "output_guidance_strength": ("output_guidance", None),
    "trajectory_guidance_strength": ("trajectory_guidance", None),
    "phase_scale": ("identity_transfer_enabled", None),
    "prompt_enhance_max_length": ("prompt_enhance", ('true',)),
    "prompt_enhance_system": ("prompt_enhance", ('true',)),
    "prompt_enhance_temperature": ("prompt_enhance", ('true',)),
    "prompt_enhance_thinking": ("prompt_enhance", ('true',)),
    "prompt_enhance_top_p": ("prompt_enhance", ('true',)),
    "score_slider_strength": ("score_slider", None),
    "source_id": ("identity_transfer_enabled", None),
    "v2a_grad_scale": ("joyai_audio_memory", None),
}


#: every knob the Editor offers, and the value it ships with. Used for the settings that
#: live inside the studio_settings blob rather than on a node, where there is no INPUT_TYPES
#: to ask. Regenerated from engine_settings.js by a test, like the table above.
_EDITOR_DEFAULTS = {
    "absolute_strength": 0.6,
    "alg_anchor": False,
    "alg_anchor_sigma_threshold": 0.975,
    "alg_anchor_strength": 2.0,
    "alg_blur_guides": False,
    "alg_guide_blur_sigma_threshold": 0.975,
    "alg_guide_blur_strength": 2.0,
    "arcface_mode": "auto_adjust",
    "bounded_attention_enabled": False,
    "carry_i2v_guides": False,
    "carry_overlap_through_anchor": False,
    "cfg": 1.0,
    "context_window_freenoise": True,
    "context_window_fuse": "pyramid",
    "context_window_length": 145,
    "context_window_overlap": 40,
    "context_window_retain_first": False,
    "context_window_schedule": "standard_uniform",
    "context_windows": False,
    "cut_opening_frames": 0,
    "debug_log": False,
    "decode_noise_scale": 0.0,
    "decode_tile_size": 0,
    "decode_timestep": 0.05,
    "detail_denoise": 0.85,
    "detail_max_area": 0.35,
    "detail_mode": "repair",
    "detail_strength": 1.0,
    "detail_targets": "hands",
    "detail_threshold": 0.35,
    "dynashift": False,
    "dynashift_strength": 0.3,
    "dynashift_threshold": 0.6,
    "embed_guidance": False,
    "embed_guidance_source": "relative",
    "embed_guidance_strength": 0.02,
    "frame_overlap": 16,
    "h3_phrase_emphasis": False,
    "h3_video_detail": 1.0,
    "id_strength": 1.0,
    "identity_transfer_enabled": False,
    "joyai_audio_memory": False,
    "joyai_fix_frames": 3,
    "joyai_frame_select": "center",
    "joyai_memory": False,
    "joyai_memory_size": 7,
    "joyai_memory_strength": 0.3,
    "negative_erase": False,
    "negative_erase_mode": "project",
    "negative_erase_renorm": True,
    "negative_erase_strength": 0.5,
    "output_guidance": False,
    "trajectory_guidance": False,
    "trajectory_guidance_strength": 0.02,
    "output_guidance_strength": 0.02,
    "phase_scale": 1.0,
    "prompt_enhance": False,
    "prompt_enhance_max_length": 400,
    "prompt_enhance_system": "",
    "prompt_enhance_temperature": 0.7,
    "prompt_enhance_thinking": False,
    "prompt_enhance_top_p": 0.92,
    "reference_injection": False,
    "score_slider": False,
    "score_slider_strength": 1.0,
    "second_pass_op": "none",
    "second_pass_upscale": 2.0,
    "segmented_detailing": False,
    "source_id": 2.0,
    "split_transition_placement": "start",
    "steer_mode": "relative",
    "taste_nearest_prompt": False,
    "temporal_style": "natural",
    "transition_duration": 16,
    "use_same_seed": False,
    "v2a_grad_scale": 1.0,
    "value_guidance": True,
    "vision_conditioning": True,
}


def _node_inputs(node_class: str) -> tuple:
    """(every input this node declares, the ones that have a default).

    Read off the NODE rather than copied here: it is the thing whose values are being
    printed, so a table in this file would be one more thing to keep in step.
    """
    try:
        from . import bridge
        spec = bridge._funpack_attr("samplers" if "Sampler" in node_class else "conditioning",
                                    node_class).INPUT_TYPES()
    except Exception:
        return frozenset(), {}
    declared, defaults = set(), {}
    for group in ("required", "optional"):
        for name, decl in (spec.get(group) or {}).items():
            declared.add(name)
            if not isinstance(decl, (list, tuple)) or not decl:
                continue
            if len(decl) >= 2 and isinstance(decl[1], dict) and "default" in decl[1]:
                defaults[name] = decl[1]["default"]
            elif isinstance(decl[0], (list, tuple)) and decl[0]:
                defaults[name] = decl[0][0]      # a combo's first entry is its default
    return frozenset(declared), defaults


def _switched_on(name: str, values: dict, defaults: dict, _seen=None) -> bool:
    """Is the feature this knob belongs to actually running?

    Walks the chain: v2a_grad_scale belongs to joyai_audio_memory, which belongs to
    joyai_memory. An owner that is itself off takes everything under it with it.
    """
    owner, wanted = _OWNED_BY.get(name, (None, None))
    if owner is None:
        return True
    seen = _seen or set()
    if owner in seen:                            # a cycle in the table is a bug, not a hang
        return True
    seen.add(owner)
    current = values.get(owner, defaults.get(owner))
    # Lowercased both sides: the Editor writes JS literals, so a boolean owner arrives as
    # "true" while Python's own str(True) is "True".
    live = bool(current) if wanted is None \
        else str(current).lower() in {str(w).lower() for w in wanted}
    return live and _switched_on(owner, values, defaults, seen)


#: keys that live on in old studio_settings blobs for features that no longer exist. There
#: is no node to check the refiner block against, so these are named rather than derived.
_DEAD_REFINER_KEYS = frozenset({"prompt_repair", "prefer_wired_conditioning"})


def _empty(value) -> bool:
    return value is None or (isinstance(value, (str, list, dict, tuple)) and len(value) == 0)


def _live_rows(values: dict, node_class: str = "", defaults: dict = None) -> list:
    """`values` filtered down to what actually decided this render.

    With a `node_class`, a key the node does not declare is DROPPED: ComfyUI builds a node's
    arguments from its own INPUT_TYPES, so a leftover key from a feature that no longer
    exists reaches nothing. `block_steer` was removed from this pack entirely and was still
    being printed as a live setting; so was a `steps` key the Chain Sampler has never had.
    """
    declared = frozenset()
    if node_class:
        declared, node_defaults = _node_inputs(node_class)
        defaults = node_defaults if defaults is None else {**defaults, **node_defaults}
    defaults = defaults or {}
    rows = []
    for name, value in (values or {}).items():
        if str(name).startswith("_"):            # the builder's own private channel
            continue
        if name in _DEAD_REFINER_KEYS:           # a feature that no longer exists
            continue
        if declared and name not in declared:    # cannot reach the render at all
            continue
        if name in defaults and value == defaults[name]:
            continue
        if _empty(value):                        # decided nothing either
            continue
        if not _switched_on(name, values, defaults):
            continue
        rows.append((name, _short(value)))
    return rows


def _sampling_sections(studio_inputs: dict, sampler_inputs: dict) -> list:
    """Everything that decided HOW it was sampled, as card sections.

    The two dictionaries are different in kind: `studio_settings` holds a full config, while
    `sampler_inputs` holds only the widgets the Editor overrode — so the first is filtered
    down to what is live and the second is printed whole.
    """
    out = []
    studio_inputs = studio_inputs or {}
    sampler_inputs = sampler_inputs or {}
    settings = {}
    raw = studio_inputs.get("studio_settings")
    if isinstance(raw, str):
        try:
            settings = json.loads(raw)
        except Exception:
            settings = {}
    elif isinstance(raw, dict):
        settings = raw

    samplers = settings.get("samplers") if isinstance(settings.get("samplers"), dict) else {}
    if samplers.get("high"):
        out.append({"title": "Sampler", "node_class": "", "rows": _pass_rows(samplers["high"])})
    # Pass 2 only when it is actually running — an off feature's settings on the card read
    # as if they were in effect.
    if sampler_inputs.get("second_pass") and samplers.get("low"):
        low = samplers["low"]
        rows = ([("own sampler", "yes")] + _pass_rows(low)) if low.get("own_sampler") else \
               [("own sampler", "no — reuses pass 1")] + _pass_rows(low)[1:2]
        out.append({"title": "Second pass", "node_class": "", "rows": rows})

    refiner = settings.get("refiner") if isinstance(settings.get("refiner"), dict) else {}
    # No node to ask here — the refiner block lives inside the studio_settings blob — so an
    # unrecognised key is KEPT. Unjudgeable is not the same as inert.
    refiner_rows = _live_rows(refiner, defaults=_EDITOR_DEFAULTS)
    if refiner_rows:
        out.append({"title": "Studio", "node_class": "", "rows": refiner_rows})

    extras = _live_rows({k: v for k, v in studio_inputs.items()
                         if k not in _SKIP_STUDIO_KEYS}, "FunPackStudio")
    if extras:
        out.append({"title": "Studio inputs", "node_class": "", "rows": extras})
    chain = _live_rows(sampler_inputs, "FunPackLTXAVSceneChainSampler")
    if chain:
        out.append({"title": "Chain Sampler", "node_class": "", "rows": chain})
    return out


def collect(models: dict, host: dict, *, project_name=None, version=None,
            codename=None, render=None, studio_inputs=None, sampler_inputs=None) -> dict:
    """The card's content, as data. Rendering is a separate step so this is testable."""
    slots = (models or {}).get("slots") or []
    sections = []
    for slot in slots:
        title = (slot.get("label") or slot.get("role_label")
                 or slot.get("node_class") or slot.get("id") or "node")
        rows = []
        for name, value in (slot.get("inputs") or {}).items():
            if name in _NOISE_INPUTS:
                continue
            if _is_wired(slot, name, slots):
                rows.append((name, "‹wired›"))
                continue
            listed = _rows_from_list(value)
            if listed is not None:
                # An empty list is one fact, not a heading with nothing under it.
                if not listed:
                    rows.append((name, "(none)"))
                    continue
                rows.append((name, ""))
                rows.extend((f"    {k}", v) for k, v in listed)
                continue
            rows.append((name, _short(value)))
        sections.append({
            "title": str(title),
            "node_class": str(slot.get("node_class") or ""),
            "rows": rows,
        })
    head = []
    if render:
        head.append({"title": "Render", "node_class": "",
                     "rows": [(k, _short(v)) for k, v in render.items()]})
    head.extend(_sampling_sections(studio_inputs, sampler_inputs))
    return {
        "project": project_name or None,
        "family": (models or {}).get("model_family") or None,
        "host": _host_rows(host),
        "sections": head + sections,
        "version": version or "",
        "codename": codename or "",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# --- rendering ------------------------------------------------------------------------

_MONO_CANDIDATES = ("DejaVuSansMono.ttf", "Menlo.ttc", "consola.ttf",
                    "LiberationMono-Regular.ttf", "CourierNew.ttf")
_SANS_CANDIDATES = ("DejaVuSans.ttf", "HelveticaNeue.ttc", "arial.ttf",
                    "LiberationSans-Regular.ttf")


def _font(size, mono=False):
    """A real font at `size`, whatever this machine happens to have.

    Pillow's bundled default is proportional; filenames read far better in mono, so try the
    usual system faces first and fall back rather than making the font a hard dependency.
    """
    from PIL import ImageFont
    for name in (_MONO_CANDIDATES if mono else _SANS_CANDIDATES):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _wrap(text, width_chars):
    """Hard-wrap on width. Model filenames have no spaces, so word wrapping alone leaves
    them running off the card."""
    text = str(text)
    if len(text) <= width_chars:
        return [text]
    out, line = [], ""
    for word in text.split(" "):
        while len(word) > width_chars:
            if line:
                out.append(line); line = ""
            out.append(word[:width_chars]); word = word[width_chars:]
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width_chars:
            line += " " + word
        else:
            out.append(line); line = word
    if line:
        out.append(line)
    return out


WIDTH = 1080
PAD = 40
LABEL_W = 260
LINE_H = 24
VALUE_CHARS = 62          # fallback when the font cannot be measured


def render_png(report: dict, theme: str = "dark") -> bytes:
    """The card as PNG bytes, with the report embedded as a tEXt chunk."""
    from PIL import Image, ImageDraw
    from PIL.PngImagePlugin import PngInfo

    pal = THEMES.get(str(theme).lower(), THEMES["dark"])
    f_title = _font(30)
    f_sub = _font(15)
    f_head = _font(16)
    f_key = _font(14, mono=True)
    f_val = _font(14, mono=True)
    f_foot = _font(13)

    # Wrap on what the value column actually holds, not on a guess: the fonts differ per
    # machine, and a fixed character count either wraps a filename that fitted or lets one
    # run off the edge.
    try:
        char_w = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength("M" * 20, font=f_val) / 20.0
        wrap_at = max(20, int((WIDTH - PAD - (PAD + LABEL_W)) / char_w)) if char_w else VALUE_CHARS
    except Exception:
        wrap_at = VALUE_CHARS

    # Lay the document out first so the canvas is exactly as tall as the content — a card
    # with a field of empty space below it looks like something failed to render.
    blocks = [("title", report.get("project") or "Models & Pipeline")]
    sub = " · ".join(x for x in (report.get("family"), report.get("generated")) if x)
    if sub:
        blocks.append(("sub", sub))
    blocks.append(("head", "Host"))
    for k, v in report.get("host") or []:
        for i, piece in enumerate(_wrap(v, wrap_at)):
            blocks.append(("row", (k if i == 0 else "", piece)))
    for section in report.get("sections") or []:
        cls = section.get("node_class")
        blocks.append(("head", f"{section['title']}   ({cls})" if cls else section["title"]))
        for k, v in section.get("rows") or []:
            if v == "":
                blocks.append(("row", (k, "")))
                continue
            for i, piece in enumerate(_wrap(v, wrap_at)):
                blocks.append(("row", (k if i == 0 else "", piece)))

    height = PAD
    for kind, _ in blocks:
        height += {"title": 44, "sub": 30, "head": 34, "row": LINE_H}[kind]
    height += 56  # footer rule + watermark

    img = Image.new("RGB", (WIDTH, height), pal["bg"])
    d = ImageDraw.Draw(img)
    y = PAD
    for kind, payload in blocks:
        if kind == "title":
            d.text((PAD, y), str(payload), font=f_title, fill=pal["text"]); y += 44
        elif kind == "sub":
            d.text((PAD, y), str(payload), font=f_sub, fill=pal["muted"]); y += 30
        elif kind == "head":
            y += 10
            d.text((PAD, y), str(payload), font=f_head, fill=pal["accent"])
            y += 24
        else:
            key, val = payload
            if key:
                d.text((PAD, y), str(key), font=f_key, fill=pal["muted"])
            d.text((PAD + LABEL_W, y), str(val), font=f_val, fill=pal["text"])
            y += LINE_H

    y += 16
    d.line([(PAD, y), (WIDTH - PAD, y)], fill=pal["line"], width=1)
    y += 12
    mark = "FunPack"
    if report.get("version"):
        mark += f" {report['version']}"
    if report.get("codename"):
        mark += f" · {report['codename']}"
    d.text((PAD, y), mark, font=f_foot, fill=pal["faint"])

    meta = PngInfo()
    try:
        meta.add_text("funpack_settings", json.dumps(report, ensure_ascii=False))
    except Exception:
        pass
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()
