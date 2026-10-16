"""Export the loaded pipeline as a PNG you can post next to a render.

"Which model was that?" is the most common question asked about a good
generation and the hardest to answer three days later. This renders the
answer as a picture: every slot's node class, its typed-in values, which
inputs are wired instead of typed, and the host facts (torch, CUDA, GPU,
attention backend) that decide whether the setup is reproducible at all on
another box.

Ported from v4's `movie_editor/backend/settings_card.py`, cut down hard: v4's
version spent most of its length (`_OWNED_BY`, `_EDITOR_DEFAULTS`,
`_pass_rows`, `_sampling_sections`, `_node_inputs`) describing FunPackStudio
and the LTX-AV Scene Chain Sampler's own hardcoded knob tables -- neither
node exists in v5's module-driven pipeline, so none of that has anything to
describe here. What is genuinely generic -- walking a slot's real inputs,
detecting a wired one, rendering the PNG -- is kept close to verbatim.

Two decisions worth knowing, both v4's:

* **Only inputs that were TYPED are listed as values.** A wired input's value
  is whatever the upstream slot produces, so printing the stale widget still
  sitting behind the socket would be a confident lie. Wired inputs are named
  as wired instead of being dropped, because "this slot has a VAE input" is
  part of the answer.
* **The JSON is embedded in the PNG's tEXt chunk** under `funpack_settings`,
  so the picture is also machine-readable. Nothing reads it back yet; it
  costs a few hundred bytes and it is the difference between a screenshot
  and a record.

Rendering is best-effort about fonts only. Everything else is derived from
the pipeline the caller hands in, so this module never has to guess.
"""
from __future__ import annotations

import io
import json
from datetime import datetime

from .graph import is_link

# Straight from styles.css, so the card matches the app it was exported from
# rather than approximating it. Only the tokens a flat document needs.
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

# This route is unauthenticated, the same as /api/pipeline -- but unlike that
# route's O(n) dict work, render_png's image height grows with wrapped-line
# count, so one crafted long value amplifies a small request into a very
# large render (found in extensive_testing: a single 200k-char value forced a
# 77,400px-tall, ~250MB image and a multi-second single-threaded draw loop).
# A value this long was never going to be readable on a fixed-width card
# anyway -- the card's whole purpose is a compact summary, not a full dump --
# so this is a real limit worth declaring rather than a corner cut short.
_MAX_VALUE_CHARS = 400
_MAX_ROWS = 2000
_MAX_KEYS_PER_LIST_ROW = 50


def _short(value):
    if isinstance(value, bool):
        return "on" if value else "off"
    if value is None or value == "":
        return "—"
    text = str(value)
    if len(text) > _MAX_VALUE_CHARS:
        return text[:_MAX_VALUE_CHARS] + f"… ({len(text) - _MAX_VALUE_CHARS} more chars)"
    return text


def _cap(value):
    """Length-cap a string that isn't a display VALUE (so no bool/"—"
    formatting) -- a slot id, group, node class, or project name. Falsy
    values pass through unchanged so "no group"/"no id" fallbacks still work.
    """
    if not value:
        return value
    text = str(value)
    if len(text) > _MAX_VALUE_CHARS:
        return text[:_MAX_VALUE_CHARS] + f"… ({len(text) - _MAX_VALUE_CHARS} more chars)"
    return text


def _rows_from_list(value):
    """A funpack_list widget's JSON array as [(label, value)], or None if it isn't one.

    Detected by content rather than by asking object_info: the value has to
    be parsed to be printed anyway, and a card that needs a live ComfyUI to
    render is a card you cannot export from a broken install.
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
    for i, row in enumerate(rows[:_MAX_ROWS], 1):
        # extensive_testing round 3: _short() on each KEY's value bounded
        # nothing about the JOINED row -- a dict with many keys produced one
        # row whose string had no cap of its own, so row count and per-key
        # length both stayed under budget while a single row's value (and
        # the report JSON embedded in the PNG's tEXt chunk, which scales
        # 1:1 with it) ballooned past 100MB. Also bound how many keys get
        # processed at all: a row with an absurd key count still costs one
        # f-string+join per key before any length cap ever applies.
        # The KEY is caller data too, same as the value -- an uncapped key
        # still gets fully materialized into `parts` before _short() on the
        # joined result ever runs, so a huge key is real per-row cost even
        # though the final string comes out short.
        items = list(row.items())
        parts = [f"{_cap(k)}={_short(v)}" for k, v in items[:_MAX_KEYS_PER_LIST_ROW]]
        if len(items) > _MAX_KEYS_PER_LIST_ROW:
            parts.append(f"…(+{len(items) - _MAX_KEYS_PER_LIST_ROW} more keys)")
        out.append((f"[{i}]", _short("  ".join(parts)) if parts else "—"))
    if len(rows) > _MAX_ROWS:
        out.append(("", f"… {len(rows) - _MAX_ROWS} more rows omitted"))
    return out          # [] for an empty list; the caller collapses that onto one line


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
        rows.append(("GPU", "none visible (CPU / MPS)" if not host.get("mps") else "Apple MPS"))
    return rows


def collect(slots: list, host: dict, *, project_name=None, render=None) -> dict:
    """The card's content, as data. Rendering is a separate step so this is testable.

    `slots` is the live pipeline exactly as `GET /api/pipeline`/`PS.slots()`
    hand it around: `{id, node, group, inputs, roles?}` per slot, a wired
    input arriving as `[source_slot_id, output_index]` (`core.graph.is_link`)
    rather than a scalar. No FunPackStudio/Chain-Sampler shape to adapt --
    v5's pipeline has no such nodes, so every slot is walked the same way.
    """
    by_id = {s.get("id"): s for s in (slots or []) if isinstance(s, dict)}
    sections = []
    total_rows = 0
    omitted_slots = 0
    mid_slot_truncations = 0
    slots_processed = 0
    for slot in slots or []:
        if not isinstance(slot, dict):
            continue
        # A card is a compact summary, not a full dump -- and without a cap
        # here, a pipeline with an unreasonable number of slots (or one slot
        # with an unreasonable number of inputs) would keep growing the PNG's
        # height with no bound (see _MAX_VALUE_CHARS's own comment for the
        # same reasoning applied to one value instead of the whole card).
        # extensive_testing round 4: a slot with no inputs contributes zero
        # rows, so total_rows alone never bounded how many SLOTS get
        # processed -- 3000 slots with empty inputs and huge id/group strings
        # reached 193.6s / 12.1MB embedded, with total_rows at 0 throughout.
        # Cap slot count directly, independent of row count.
        if total_rows >= _MAX_ROWS or slots_processed >= _MAX_ROWS:
            omitted_slots += 1
            continue
        slots_processed += 1
        # Same round: slot id/group/node were never length-capped at all --
        # they reach the "head" block kind in render_png, which the round-2
        # line-truncation loop doesn't count (only "row" kind), so an
        # uncapped id/group bypassed every existing cap.
        group = _cap(slot.get("group"))
        node = _cap(slot.get("node") or "")
        slot_id = _cap(slot.get("id"))
        title = f"{group} · {slot_id}" if group else str(slot_id or node or "node")
        rows = []
        # extensive_testing round 2: this check used to run once per SLOT, so
        # one slot with many funpack_list inputs (each individually capped at
        # _MAX_ROWS by _rows_from_list) could blow the card-wide budget by a
        # large multiple before the *next* slot's check ever fired. Checking
        # the running budget per INPUT closes that.
        for name, value in (slot.get("inputs") or {}).items():
            if total_rows + len(rows) >= _MAX_ROWS:
                mid_slot_truncations += 1
                break
            if name in _NOISE_INPUTS:
                continue
            # `inputs` is a caller-controlled dict -- the KEY, not just the
            # value, is attacker data. Every row below carries `name`
            # verbatim into the drawn label and the embedded JSON.
            name = _cap(name)
            if is_link(value):
                source = by_id.get(value[0])
                source_label = _cap((source or {}).get("id") or value[0])
                rows.append((name, f"‹wired from {source_label}›"))
                continue
            listed = _rows_from_list(value)
            if listed is not None:
                if not listed:
                    rows.append((name, "(none)"))
                    continue
                rows.append((name, ""))
                rows.extend((f"    {k}", v) for k, v in listed)
                continue
            rows.append((name, _short(value)))
        total_rows += len(rows)
        sections.append({"title": title, "node_class": node, "rows": rows})
    if omitted_slots or mid_slot_truncations:
        parts = []
        if mid_slot_truncations:
            parts.append(f"{mid_slot_truncations} slot(s) cut short")
        if omitted_slots:
            parts.append(f"{omitted_slots} slot(s) omitted entirely")
        sections.append({"title": "(truncated)", "node_class": "",
                         "rows": [("", f"… {', '.join(parts)} "
                                       f"— {_MAX_ROWS}-row card limit reached")]})

    head = []
    if render:
        head.append({"title": "Render", "node_class": "",
                     "rows": [(k, _short(v)) for k, v in render.items()]})
    return {
        "project": _cap(project_name) or None,
        "host": _host_rows(host),
        "sections": head + sections,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# --- rendering ------------------------------------------------------------------------

_MONO_CANDIDATES = ("DejaVuSansMono.ttf", "Menlo.ttc", "consola.ttf",
                    "LiberationMono-Regular.ttf", "CourierNew.ttf")
_SANS_CANDIDATES = ("DejaVuSans.ttf", "HelveticaNeue.ttc", "arial.ttf",
                    "LiberationSans-Regular.ttf")


def _font(size, mono=False):
    """A real font at `size`, whatever this machine happens to have.

    Pillow's bundled default is proportional; filenames read far better in
    mono, so try the usual system faces first and fall back rather than
    making the font a hard dependency.
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
    """Hard-wrap on width. Model filenames have no spaces, so word wrapping
    alone leaves them running off the card."""
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
    f_val = _font(14, mono=True)
    f_key = f_val
    f_foot = _font(13)

    try:
        char_w = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength("M" * 20, font=f_val) / 20.0
        wrap_at = max(20, int((WIDTH - PAD - (PAD + LABEL_W)) / char_w)) if char_w else VALUE_CHARS
    except Exception:
        wrap_at = VALUE_CHARS

    # extensive_testing round 5: row VALUES are wrapped to card width before
    # drawing (bounding each d.text() call's cost), but "head"/"title" block
    # payloads were drawn as one unwrapped string -- PIL's text-render cost
    # scales with string length regardless of visible width, so a _cap()
    # -allowed 400-char title, repeated once per slot up to _MAX_ROWS slots,
    # cost 20.5s of pure font-render CPU even though every documented cap
    # was individually honored. Single-line draws still need a width bound.
    _HEAD_DRAW_CHARS = wrap_at

    def _draw_cap(text):
        text = str(text)
        return text[:_HEAD_DRAW_CHARS] + "…" if len(text) > _HEAD_DRAW_CHARS else text

    blocks = [("title", _draw_cap(report.get("project") or "FunPack pipeline"))]
    sub = report.get("generated")
    if sub:
        blocks.append(("sub", sub))
    blocks.append(("head", "Host"))
    for k, v in report.get("host") or []:
        for i, piece in enumerate(_wrap(v, wrap_at)):
            blocks.append(("row", (k if i == 0 else "", piece)))
    for section in report.get("sections") or []:
        cls = section.get("node_class")
        head_text = f"{section['title']}   ({cls})" if cls else section["title"]
        blocks.append(("head", _draw_cap(head_text)))
        for k, v in section.get("rows") or []:
            if v == "":
                blocks.append(("row", (k, "")))
                continue
            for i, piece in enumerate(_wrap(v, wrap_at)):
                blocks.append(("row", (k if i == 0 else "", piece)))

    # extensive_testing round 2: _MAX_ROWS and _MAX_VALUE_CHARS each bound one
    # factor, but a value near the char cap still wraps into several lines
    # (no spaces to break on), so their PRODUCT was never bounded -- 2000
    # rows of 400-char values honored both caps individually and still cost
    # ~14s/~10MB. Cap the thing that actually drives render_png's cost --
    # rendered lines -- directly, rather than trying to predict it from
    # collect()'s row count.
    # extensive_testing round 4: this loop only counted "row" kind blocks,
    # so slot count itself was unbounded whenever slots carried no rows (an
    # empty-input slot contributes only a "head" block) -- 3000 such slots
    # reached 193.6s/12.1MB before slot count was separately capped in
    # collect(). Count "head" blocks toward the same budget too, as defense
    # in depth against any other way a block could reach render_png with no
    # accompanying row.
    _MAX_WRAPPED_LINES = 1200
    row_lines = 0
    capped_blocks = []
    line_truncated = False
    for kind, payload in blocks:
        if kind in ("row", "head"):
            if row_lines >= _MAX_WRAPPED_LINES:
                line_truncated = True
                continue
            row_lines += 1
        capped_blocks.append((kind, payload))
    if line_truncated:
        capped_blocks.append(("head", "(truncated)"))
        capped_blocks.append(("row", ("", f"… card exceeded {_MAX_WRAPPED_LINES} "
                                           f"rendered lines, rest omitted")))
    blocks = capped_blocks

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
    d.text((PAD, y), "FunPack", font=f_foot, fill=pal["faint"])

    meta = PngInfo()
    try:
        meta.add_text("funpack_settings", json.dumps(report, ensure_ascii=False))
    except Exception:
        pass
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()
