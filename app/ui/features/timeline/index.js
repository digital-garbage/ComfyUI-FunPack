// The cutting-room timeline. One hub, several features: the stage (clips you trim, reorder, split), the buttons
// that act on the cut, the zoom controls, and the count in the zone's head. They share only the app's
// selection, playhead and message bus.
import { composer as c } from "../../composer/composer.js";
import { segments, totalSeconds, clock } from "../../shell/scenes.js";
import { inPlace } from "../../shell/place.js";
import * as edits from "../../shell/edits.js";
import { moveLane, trimLane } from "../../shell/audio.js";
import { videoLane, audioLane, tracksLane, framesAt } from "./lanes.js";

const ZOOM = [20, 40, 80, 160, 320];
const sceneIds = (p) => segments(p).filter((s) => s.kind === "scene").map((s) => s.id);

const stage = {
  id: "timeline",
  mount: "timeline",
  needs: ["project", "selection", "playhead"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection, head = app.playhead;
    let zoom = 2;
    const edit = (fn) => p.edit(fn);
    const extra = app.timelineLanes || [];       // lanes other features add: { owns(id), lanes(project), select/move/trim(id, ...) }
    const owner = (id) => extra.find((e) => e.owns(id));
    const view = c.timeline.stage({
      pxPerSecond: ZOOM[zoom],
      onSeek: (sec) => head.set(sec),
      onSelect: (id, how) => { const own = owner(id); if (own) own.select(id); else sel.pick(id, how, sceneIds(p.project)); },
      onMove: (id, d) => { const own = owner(id); if (own) own.move(id, d); else id.startsWith("t:") && edit((pr) => moveLane(pr, id.slice(2), d)); },
      onTrim: (id, edge, d) => {
        const own = owner(id);
        if (own) return own.trim(id, edge, d);
        if (id.startsWith("t:")) return edit((pr) => trimLane(pr, id.slice(2), edge, d));
        const dur = segments(p.project).find((s) => s.id === id).dur;
        edit((pr) => edge === "out" ? edits.resize(pr, id, dur + d) : d > 0 && edits.trimLeft(pr, id, d));   // the left edge only cuts in
      },
      onReorder: (id, index) => edit((pr) => {      // the stage counts ghosts and pauses too; the cut counts scenes
        const all = segments(pr).filter((s) => s.id !== id);
        return edits.reorder(pr, id, all.slice(0, index).filter((s) => s.kind === "scene").length, sceneIds(pr));
      }),
    });
    const move = (id, by) => edit((pr) => { const order = sceneIds(pr), at = order.indexOf(id); return at >= 0 && edits.reorder(pr, id, at + by, order); });
    const clipActions = (sc) => [{ label: "◀", title: "Move left in the cut", onClick: () => move(sc.id, -1) }, { label: "▶", title: "Move right in the cut", onClick: () => move(sc.id, 1) },
      { label: "Remove", title: "Remove this clip", onClick: () => edit((pr) => edits.removeScene(pr, sc.id)) }];
    const ghostActions = (g) => [{ label: "✕", title: "Remove from the timeline",
      onClick: () => edit((pr) => { pr.scene_ghosts = (pr.scene_ghosts || []).filter((x) => x.id !== g.id); return true; }) }];

    const draw = () => { const open = p.project; view.setLanes(open ? [videoLane(open, sel.ids, sel.focus, clipActions, ghostActions), audioLane(open, sel.ids), tracksLane(open), ...extra.flatMap((e) => { try { return e.lanes(open); } catch { return []; } })]       // a feature's lanes failing must not stop the baseline's.filter(Boolean) : []); };
    const setZoom = (z) => { zoom = Math.min(ZOOM.length - 1, Math.max(0, z)); view.setZoom(ZOOM[zoom]); };
    host.append(view.node);
    const off = [app.on((what) => {
      if (what === "zoom.in") setZoom(zoom + 1); else if (what === "zoom.out") setZoom(zoom - 1);
      else if (what === "zoom.fit") { const total = p.project ? totalSeconds(p.project) : 0; const fit = ZOOM.filter((z) => z * (total + 2) <= view.node.clientWidth); setZoom(fit.length ? ZOOM.indexOf(fit[fit.length - 1]) : 0); }
      else draw();
    }), head.on((sec) => view.setPlayhead(sec))];
    draw();
    return () => { off.forEach((f) => f()); view.destroy(); };
  },
};

// Add / Split / Remove, the timecode, and how many clips are picked.
const tools = {
  id: "timeline-tools",
  mount: "timeline.toolbar",
  needs: ["project", "selection", "playhead"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection, head = app.playhead;
    const put = inPlace(host);
    const btn = (label, onClick, disabled, title) => c.button.sm({ label, onClick, disabled, title, tone: "ghost" }).node;
    const split = () => {       // the selected clip, at the playhead if it is over it, else in the middle
      if (!p.project) return;
      const seg = segments(p.project).find((s) => s.id === sel.focus && s.kind === "scene");
      if (!seg) return;
      const inside = head.at > seg.start && head.at < seg.start + seg.dur;
      if (!p.edit((pr) => edits.split(pr, seg.id, inside ? framesAt(seg.scene, pr, head.at - seg.start) : undefined))) c.toast.warn({ text: "Too close to the clip's edge to cut there." });
    };
    const draw = () => {
      const open = p.project, ids = open ? sel.ids : [];
      put(c.text.sm({ text: `${clock(head.at)} / ${clock(open ? totalSeconds(open) : 0)}` }).node,
        btn("＋ Add", () => { const sc = p.edit((pr) => edits.addScene(pr)); if (sc) p.select(sc.id); }, !open),
        btn("Split", split, !ids.length, "Split the selected clip at the playhead"),
        btn("Remove", () => ids.forEach((id) => p.edit((pr) => edits.removeScene(pr, id))), !ids.length),
        c.chip.neutral({ label: ids.length ? `${ids.length} clip${ids.length > 1 ? "s" : ""} selected` : "no clip selected" }).node);
    };
    draw();
    const off = [app.on((what) => (what === "split" ? split() : draw())), head.on(draw)];
    return () => off.forEach((f) => f());
  },
};

// The hint and the zoom controls, at the end of the same row.
const zoom = {
  id: "timeline-zoom",
  mount: "timeline.toolbar.end",
  needs: ["say"],
  setup({ host, app }) {
    const b = (label, what, title) => c.button.sm({ label, tone: "ghost", title, onClick: () => app.say(what) }).node;
    host.append(c.text.sm({ text: "J/K/L · S split · I/O in/out · +/- zoom" }).node, c.text.sm({ text: "zoom" }).node,
      b("−", "zoom.out", "Zoom out"), b("＋", "zoom.in", "Zoom in"), b("fit", "zoom.fit", "Fit the cut to the window"));
  },
};

// "4 clips · 4 active · 1 selected · 00:09:19" at the far end of the zone's head.
const count = {
  id: "timeline-count",
  mount: "timeline.status",
  needs: ["project", "selection"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection;
    const put = inPlace(host);
    const draw = () => {
      const open = p.project, scenes = open ? open.scenes : [];
      put(c.text.sm({ text: `${scenes.length} clips · ${scenes.filter((s) => !s.excluded).length} active · ${open ? sel.ids.length : 0} selected · ${clock(open ? totalSeconds(open) : 0)}` }).node);
    };
    draw();
    return app.on(draw);
  },
};

export default [stage, tools, zoom, count];
