// The window that makes or edits one overlay. Nothing changes in the project until Save / Add.
import { composer as c } from "../../composer/composer.js";
import { FONTS, TEXT_DEFAULTS } from "../../shell/overlays.js";

const field = (label, control, hint) => c.field.default({ label, hint, control });
const options = (list) => list.map((v) => ({ value: v, label: v[0].toUpperCase() + v.slice(1) }));

function textForm(s) {
  const num = (label, key, min, max, step) => field(label, c.number.md({ label, value: s[key], min, max, step, onChange: (v) => { s[key] = v; } }));
  const tog = (label, key) => c.toggle.default({ label, checked: Boolean(s[key]), onChange: (v) => { s[key] = v; } });
  const col = (label, key) => field(label, c.color.swatch({ label, value: s[key], onChange: (v) => { s[key] = v; } }));
  return [
    field("Text", c.textarea.md({ label: "Text", value: s.text, rows: 3, onInput: (v) => { s.text = v; } })),
    num("Font size (px)", "font_size", 8, 400, 1),
    field("Font", c.select.md({ label: "Font", value: s.font_family, options: options(FONTS), onChange: (v) => { s.font_family = v; } })),
    col("Colour", "color"),
    field("Opacity", c.slider.readout({ label: "Opacity", value: s.opacity, min: 0, max: 1, step: 0.05, precision: 2, onCommit: (v) => { s.opacity = v; } })),
    tog("Bold", "bold"), tog("Italic", "italic"),
    field("Alignment", c.segmented.md({ label: "Alignment", value: s.text_align, options: options(["left", "center", "right"]), onChange: (v) => { s.text_align = v; } })),
    num("Line spacing", "line_spacing", 0.8, 3, 0.1),
    num("Outline width (px)", "stroke_width", 0, 40, 1), col("Outline colour", "stroke_color"),
    tog("Shadow", "shadow"), col("Shadow colour", "shadow_color"),
    tog("Background box", "bg_enabled"), col("Box colour", "bg_color"),
    field("Box opacity", c.slider.readout({ label: "Box opacity", value: s.bg_opacity, min: 0, max: 1, step: 0.05, precision: 2, onCommit: (v) => { s.bg_opacity = v; } })),
  ];
}

function imageForm(s, bin) {
  return [
    field("Picture", c.select.md({ label: "Picture", value: s.media_ref || "", options: [{ value: "", label: "Choose a picture…" }, ...(s.media_ref && !bin.some((m) => m.id === s.media_ref) ? [{ value: s.media_ref, label: "(no longer in the media bin)" }] : []), ...bin.map((m) => ({ value: m.id, label: m.name }))],
      onChange: (v) => { s.media_ref = v; s.label = (bin.find((m) => m.id === v) || {}).name || "Image"; } }), bin.length ? undefined : "The media bin has no pictures yet."),
    field("Width (px)", c.number.md({ label: "Width", value: s.width_px, min: 8, step: 1, precision: 0, onChange: (v) => { s.width_px = v; } })),
    c.toggle.default({ label: "Keep the picture's proportions", checked: s.keep_aspect !== false, onChange: (v) => { s.keep_aspect = v; if (!v && s.height_px == null) s.height_px = s.width_px; } }),
    field("Height (px)", c.number.md({ label: "Height", value: s.height_px ?? s.width_px, min: 8, step: 1, precision: 0, onChange: (v) => { s.height_px = v; } }), "Used only when the proportions are not kept."),
    field("Opacity", c.slider.readout({ label: "Opacity", value: s.opacity ?? 1, min: 0, max: 1, step: 0.05, precision: 2, onCommit: (v) => { s.opacity = v; } })),
    c.toggle.default({ label: "Flip left–right", checked: Boolean(s.flip_h), onChange: (v) => { s.flip_h = v; } }),
    c.toggle.default({ label: "Flip upside down", checked: Boolean(s.flip_v), onChange: (v) => { s.flip_v = v; } }),
  ];
}

/** kind "text"|"image", `existing` an overlay (edit) or null (new), `bin` the pictures to choose from.
 *  `onSave(patch)` runs once with the chosen values; the window closes after it. */
export function openDialog({ kind, existing, bin = [], projectWidth = 768, onSave }) {
  const s = kind === "text" ? { ...TEXT_DEFAULTS, text: "Title", ...(existing || {}) } : { width_px: Math.round(projectWidth * 0.35), keep_aspect: true, opacity: 1, ...(existing || {}) };
  const where = field("Starts at (s)", c.number.md({ label: "Starts at", value: existing ? existing.start_sec : undefined, min: 0, step: 0.1, onChange: (v) => { s.start_sec = v; } }));
  const dur = field("Shown for (s)", c.number.md({ label: "Shown for", value: s.duration_sec ?? 3, min: 0.1, step: 0.1, onChange: (v) => { s.duration_sec = v; } }));
  const body = c.region.stack({ gap: "sm", children: [...(kind === "text" ? textForm(s) : imageForm(s, bin)), ...(existing ? [where, dur] : [])] });
  const save = () => {
    if (kind === "image" && !s.media_ref) return c.toast.warn({ text: "Choose a picture first." });
    if (kind === "text") s.text = String(s.text || "").trim() || "Title";       // what the render would draw anyway
    onSave(s);
    win.close("done");
  };
  const win = c.modal.generic({ title: `${existing ? "Edit" : "Add"} ${kind} overlay`, size: "md", body });
  win.setFooter({ actions: [c.button.sm({ label: "Cancel", tone: "ghost", onClick: () => win.close("cancel") }), c.button.sm({ label: existing ? "Save" : "Add", tone: "primary", onClick: save })] });
  return win;
}
