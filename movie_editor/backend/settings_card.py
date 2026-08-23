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
    if refiner:
        out.append({"title": "Studio", "node_class": "",
                    "rows": [(k, _short(v)) for k, v in refiner.items()]})

    extras = {k: v for k, v in studio_inputs.items() if k not in _SKIP_STUDIO_KEYS}
    if extras:
        out.append({"title": "Studio inputs", "node_class": "",
                    "rows": [(k, _short(v)) for k, v in extras.items()]})
    if sampler_inputs:
        out.append({"title": "Chain Sampler", "node_class": "",
                    "rows": [(k, _short(v)) for k, v in sampler_inputs.items()]})
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
