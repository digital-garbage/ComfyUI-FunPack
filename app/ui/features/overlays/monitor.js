// Overlays on the monitor: what is shown at the playhead, drawn over the picture where the render will put it; drag one to move it.
import { composer as c } from "../../composer/composer.js";
import * as ov from "../../shell/overlays.js";

/** An overlay of the project as the kit's stage draws it: plain data, sizes in the picture's own pixels. */
function asItem(o, open) {
  const base = { id: o.id, x: o.x, y: o.y, flipH: o.flip_h, flipV: o.flip_v, opacity: o.opacity };
  if (o.kind === "text") {
    return { ...base, kind: "text", text: o.text, fontSize: o.font_size, color: o.color, family: o.font_family, bold: o.bold, italic: o.italic, align: o.text_align,
      lineSpacing: o.line_spacing, shadow: o.shadow, shadowColor: o.shadow_color, strokeWidth: o.stroke_width, strokeColor: o.stroke_color,
      bg: o.bg_enabled ? { color: o.bg_color, opacity: o.bg_opacity } : null };
  }
  const cw = open.width || 768, wpx = o.width_px != null ? o.width_px : Math.max(8, Math.min(1.5, Math.max(0.05, o.scale ?? 0.35)) * cw);
  return { ...base, kind: "image", src: `/funpack/api/media/${encodeURIComponent(o.media_ref || "")}/file`, widthPx: wpx, heightPx: o.keep_aspect === false ? (o.height_px || wpx) : null };
}

export default {
  id: "overlay-monitor",
  mount: "preview",
  needs: ["project", "playhead"],
  setup({ host, app }) {
    const p = app.project, head = app.playhead;
    let drawn = "";
    const shownAt = (open) => ov.inStackOrder(open).filter((o) => head.at >= (o.start_sec || 0) && head.at <= (o.start_sec || 0) + (o.duration_sec || 0));
    const again = () => { drawn = ""; draw(); };
    const stage = c.stage.overlay({
      onMove: (id, to) => p.edit((pr) => ov.update(pr, id, to)),
      onResize: (id, f) => {       // a picture grows by its width, a text by its size
        const o = ov.find(p.project, id), open = p.project;
        if (!o) return;
        const item = asItem(o, open);
        p.edit((pr) => ov.update(pr, id, o.kind === "text" ? { font_size: Math.max(8, Math.round((o.font_size || 42) * f)) }
          : { width_px: Math.max(8, Math.round(item.widthPx * f)), ...(item.heightPx ? { height_px: Math.max(8, Math.round(item.heightPx * f)) } : {}) }));
      },
      onCancel: again,
      onLayout: () => draw(),
    });
    host.append(stage.node);

    function draw() {
      if (stage.busy) return;
      const open = p.project;
      if (!open || !stage.fit(open.width || 768, open.height || 512)) { stage.clear(); drawn = ""; return; }
      const shown = shownAt(open), { w, h } = stage.size;
      const key = JSON.stringify([shown, w, h, open.width, open.height]);       // the playhead ticks every frame: rebuild only when what is drawn changes
      if (key === drawn) return;
      drawn = key;
      stage.setItems(shown.map((o) => asItem(o, open)));
    }
    const off = [head.on(draw), app.on(draw)];
    draw();
    return () => { off.forEach((f) => f()); stage.destroy(); };
  },
};
