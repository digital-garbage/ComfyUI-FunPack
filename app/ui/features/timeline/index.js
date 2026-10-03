// The cutting-room timeline: scenes as clips on a stage, trimmed, reordered, split, added and removed by hand.
import { composer as c } from "../../composer/composer.js";
import { createSelection } from "../../shell/selection.js";
import { segments, totalSeconds, clock } from "../../shell/scenes.js";
import * as edits from "../../shell/edits.js";
import { videoLane, framesAt } from "./lanes.js";

const ZOOM = [20, 40, 80, 160, 320];

export default {
  id: "timeline",
  mount: "timeline",
  needs: ["project"],
  setup({ host, app }) {
    const p = app.project;
    const sel = createSelection({ project: p });
    let zoom = 2;
    let at = 0;

    const edit = (fn) => p.edit(fn);
    const stage = c.timeline.stage({
      pxPerSecond: ZOOM[zoom],
      onSeek: (sec) => { at = sec; stage.setPlayhead(sec); bar(); },
      onSelect: (id, how) => sel.pick(id, how, segments(p.project).filter((s) => s.kind === "scene").map((s) => s.id)),
      onTrim: (id, edge, d) => {
        const sc = p.project.scenes.find((s) => s.id === id);
        if (!sc) return;
        const dur = segments(p.project).find((s) => s.id === id).dur;
        edit((pr) => edge === "out" ? edits.resize(pr, id, dur + d) : d > 0 && edits.trimLeft(pr, id, d));   // the left edge only cuts in
      },
      onReorder: (id, index) => edit((pr) => {      // the stage counts ghosts and pauses too; the cut counts scenes
        const all = segments(pr).filter((s) => s.id !== id);
        const order = segments(pr).filter((s) => s.kind === "scene").map((s) => s.id);
        return edits.reorder(pr, id, all.slice(0, index).filter((s) => s.kind === "scene").length, order);
      }),
    });

    let toolbar = c.toolbar.default({ items: [] });
    let zoomBar = c.toolbar.default({ items: [] });
    const swap = (old, items) => { const next = c.toolbar.default({ items }); old.node.replaceWith(next.node); return next; };

    const btn = (label, onClick, disabled = false, title) => c.button.sm({ label, onClick, disabled, title });
    function bar() {
      const open = p.project, ids = open ? sel.ids : [];
      const splitAt = () => {       // the selected clip, at the playhead if it is over it, else in the middle
        const seg = segments(open).find((s) => s.id === sel.focus && s.kind === "scene");
        if (!seg) return;
        const inside = at > seg.start && at < seg.start + seg.dur;
        if (!edit((pr) => edits.split(pr, seg.id, inside ? framesAt(seg.scene, pr, at - seg.start) : undefined))) c.toast.warn({ text: "Too close to the clip's edge to cut there." });
      };
      toolbar = swap(toolbar, [
        c.text.sm({ text: `${clock(at)} / ${clock(open ? totalSeconds(open) : 0)}` }),
        btn("＋ Add", () => { const sc = edit((pr) => edits.addScene(pr)); if (sc) p.select(sc.id); }, !open),
        btn("Split", splitAt, !ids.length, "Split the selected clip at the playhead"),
        btn("Remove", () => ids.forEach((id) => edit((pr) => edits.removeScene(pr, id))), !ids.length),
        ids.length > 1 ? c.chip.neutral({ label: `${ids.length} selected` }) : c.text.sm({ text: "" }),
      ]);
      zoomBar = swap(zoomBar, [
        c.text.sm({ text: "zoom" }),
        btn("−", () => setZoom(zoom - 1), zoom <= 0), btn("+", () => setZoom(zoom + 1), zoom >= ZOOM.length - 1),
      ]);
    }
    function setZoom(z) { zoom = Math.min(ZOOM.length - 1, Math.max(0, z)); stage.setZoom(ZOOM[zoom]); bar(); }

    const draw = () => {
      const open = p.project;
      stage.setLanes(open ? [videoLane(open, sel.ids, sel.focus, undefined, (g) => [{ label: "✕", title: "Remove from the timeline",
        onClick: () => edit((pr) => { pr.scene_ghosts = (pr.scene_ghosts || []).filter((x) => x.id !== g.id); return true; }) }])] : []);
      bar();
    };
    host.append(toolbar.node, zoomBar.node, stage.node);
    const off = [app.on(draw), sel.on(draw)];
    draw();
    return () => { off.forEach((f) => f()); stage.destroy(); };
  },
};
