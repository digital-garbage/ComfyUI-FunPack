// Overlays on the monitor: what is shown at the playhead, drawn over the picture where the render will put it; drag one to move it.
import * as ov from "../../shell/overlays.js";
import { drag } from "../../composer/internals/drag.js";

export default {
  id: "overlay-monitor",
  mount: "preview",
  needs: ["project", "playhead"],
  setup({ host, app }) {
    const p = app.project, head = app.playhead;
    const layer = document.createElement("div");
    layer.style.cssText = "position:absolute;pointer-events:none;overflow:hidden;z-index:2";
    host.style.position = "relative";
    host.append(layer);
    let box = { w: 1, h: 1 }, dragging = false, drawn = "";

    function place() {       // the picture is the project's aspect, centred in the monitor (or in its "no render" panel)
      const open = p.project, area = [...host.children].find((n) => n !== layer && !n.hidden && n.getBoundingClientRect().height > 60);
      if (!area || !open) return false;
      const s = area.getBoundingClientRect(), h0 = host.getBoundingClientRect(), pw = open.width || 768, ph = open.height || 512;
      const k = Math.min(s.width / pw, s.height / ph);
      box = { w: pw * k, h: ph * k, k };
      Object.assign(layer.style, { left: `${s.left - h0.left + (s.width - box.w) / 2}px`, top: `${s.top - h0.top + (s.height - box.h) / 2}px`, width: `${box.w}px`, height: `${box.h}px` });
      return box.w > 0 && box.h > 0;
    }

    function node(o, open) {
      const el = document.createElement("div");
      el.style.cssText = `position:absolute;left:${(o.x ?? 0.5) * 100}%;top:${(o.y ?? 0.5) * 100}%;transform:translate(-50%,-50%) scale(${o.flip_h ? -1 : 1},${o.flip_v ? -1 : 1});opacity:${o.opacity ?? 1};pointer-events:auto;cursor:move;touch-action:none;`;
      el.dataset.overlay = o.id;
      if (o.kind === "text") {
        const px = (o.font_size || 42) * box.k;
        el.textContent = String(o.text || "").trim() || "Title";       // the render trims it too
        Object.assign(el.style, { whiteSpace: "pre", fontSize: `${px}px`, color: o.color || "#fff", fontFamily: o.font_family || "arial", fontWeight: o.bold ? "700" : "400", fontStyle: o.italic ? "italic" : "normal",
          textAlign: o.text_align || "center", lineHeight: String(o.line_spacing || 1.2),
          textShadow: o.shadow ? `${px * 0.06}px ${px * 0.06}px 0 ${o.shadow_color || "#000"}` : "none",
          webkitTextStroke: o.stroke_width ? `${o.stroke_width * box.k}px ${o.stroke_color || "#000"}` : "0" });
        if (o.bg_enabled) Object.assign(el.style, { background: `color-mix(in srgb, ${o.bg_color || "#000"} ${Math.round((o.bg_opacity ?? 0.5) * 100)}%, transparent)`, padding: `${px * 0.22}px ${px * 0.4}px`, borderRadius: `${px * 0.1}px` });
      } else {
        const img = document.createElement("img");
        img.src = `/funpack/api/media/${encodeURIComponent(o.media_ref || "")}/file`;
        img.draggable = false;
        const cw = open.width || 768, wpx = o.width_px != null ? o.width_px : Math.max(8, Math.min(1.5, Math.max(0.05, o.scale ?? 0.35)) * cw), w = wpx * box.k;
        img.style.cssText = `width:${w}px;${o.keep_aspect === false ? `height:${(o.height_px || wpx) * box.k}px;` : ""}display:block;`;
        el.append(img);
      }
      let to = null;
      drag(el, {
        onStart: () => { dragging = true; },
        onMove: ({ dx, dy }) => { to = { x: Math.min(1, Math.max(0, (o.x ?? 0.5) + dx / box.w)), y: Math.min(1, Math.max(0, (o.y ?? 0.5) + dy / box.h)) }; el.style.left = `${to.x * 100}%`; el.style.top = `${to.y * 100}%`; },
        onEnd: ({ cancelled }) => { dragging = false; if (to && !cancelled) p.edit((pr) => ov.update(pr, o.id, to)); else draw(); },
      });
      return el;
    }

    function draw() {
      if (dragging) return;
      const open = p.project;
      if (!open || !place()) { layer.replaceChildren(); drawn = ""; return; }
      const at = head.at, shown = ov.inStackOrder(open).filter((o) => at >= (o.start_sec || 0) && at <= (o.start_sec || 0) + (o.duration_sec || 0));
      const key = JSON.stringify([shown, box.w, box.h, open.width, open.height]);       // the playhead ticks every frame: rebuild only when what is drawn changes
      if (key === drawn) return;
      drawn = key;
      layer.replaceChildren(...shown.map((o) => node(o, open)));
    }
    const ro = new ResizeObserver(draw);
    ro.observe(host);
    const off = [head.on(draw), app.on(draw)];
    draw();
    return () => { off.forEach((f) => f()); ro.disconnect(); layer.remove(); };
  },
};
