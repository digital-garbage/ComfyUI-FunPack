// Overlays: text and pictures laid over the cut. Adds lanes to the timeline (through app.timelineLanes) and a "＋ Overlay" button.
import { composer as c } from "../../composer/composer.js";
import * as ov from "../../shell/overlays.js";
import { learn } from "../../shell/bin.js";
import { onMediaDrop } from "../../shell/dnd.js";
import { openDialog } from "./dialog.js";

const ID = "o:";
const short = (o) => String((o.kind === "text" ? o.text : o.label) || "Overlay").split("\n")[0].slice(0, 40);

export default {
  id: "overlays",
  mount: "timeline.toolbar",
  needs: ["project", "playhead", "api"],
  setup({ host, app }) {
    const p = app.project;
    const edit = (fn) => p.edit(fn);
    const pictures = async () => { const list = (await app.api.media()).media || []; learn(list); return list.filter((m) => m.kind === "image"); };

    async function open(kind, existing) {
      let bin = [];
      if (kind === "image") { try { bin = await pictures(); } catch (err) { return c.toast.warn({ text: `Could not read the media bin: ${err.message}` }); } }
      openDialog({ kind, existing, bin, projectWidth: p.project ? p.project.width : undefined, onSave: (s) => edit((pr) => {
        if (existing) return ov.update(pr, existing.id, s);
        const at = app.playhead.at, made = kind === "text" ? ov.addText(pr, at, s) : ov.addImage(pr, s.media_ref, s.label, at);
        return ov.update(pr, made.id, s);
      }) });
    }
    const confirmRemove = async (o) => { if (await c.modal.dialogue({ title: "Remove overlay", message: `Remove “${short(o)}” from the cut?`, tone: "danger", confirmLabel: "Remove" }).result) edit((pr) => ov.remove(pr, o.id)); };

    const entry = {
      owns: (id) => id.startsWith(ID),
      lanes(pr) {
        const known = new Set(ov.lanesOf(pr).map((l) => l.id));
        const lanes = [...(ov.overlaysOf(pr).some((o) => !known.has(o.lane_id)) ? [{ id: "", label: "Unplaced" }] : []), ...ov.lanesOf(pr)];       // a v4 project's overlays without a lane still show, at the bottom as they render
        return [...lanes].reverse().map((lane) => ({ id: `ov:${lane.id}`, label: lane.label, kind: "overlay", move: true,
          clips: ov.overlaysOf(pr).filter((o) => (lane.id ? o.lane_id === lane.id : !known.has(o.lane_id))).map((o) => ({ id: ID + o.id, start: o.start_sec || 0, dur: o.duration_sec || 0.1, trim: true, title: short(o),
            head: [o.kind === "text" ? "T" : "▣"],
            actions: [{ label: "✎", title: "Edit", onClick: () => open(o.kind, o) }, { label: "▲", title: "Draw above the others (higher lane)", onClick: () => edit((x) => ov.restack(x, o.id, 1)) },
              { label: "▼", title: "Draw below the others (lower lane)", onClick: () => edit((x) => ov.restack(x, o.id, -1)) }, { label: "✕", title: "Remove", onClick: () => confirmRemove(o) }] })) }));
      },
      move: (id, d) => edit((pr) => ov.move(pr, id.slice(ID.length), d)),
      trim: (id, edge, d) => edit((pr) => ov.trim(pr, id.slice(ID.length), edge, d)),
      select: (id) => { const o = ov.find(p.project, id.slice(ID.length)); if (o) open(o.kind, o); },
    };
    app.timelineLanes.push(entry);
    app.say("timeline.lanes");          // the stage draws what it has; tell it there is more

    const add = c.button.menu({ label: "＋ Overlay", tone: "ghost", disabled: !p.project, onClick: () => c.menu.dropdown({ anchor: add,
      items: [{ id: "text", label: "Text…" }, { id: "image", label: "Picture…" }, { separator: true }, { id: "lane", label: "New lane on top" },
        { id: "unlane", label: "Remove top lane", danger: true, disabled: !ov.lanesOf(p.project).length }],
      onPick: (id) => {
        if (id === "text" || id === "image") return open(id);
        if (id === "lane") return edit((pr) => ov.addLane(pr));
        const top = ov.lanesOf(p.project).slice(-1)[0];
        if (!top) return;
        if (top && !ov.overlaysOf(p.project).some((o) => o.lane_id === top.id)) return edit((pr) => ov.removeLane(pr, top.id));
        c.modal.dialogue({ title: "Remove lane", message: `Remove “${top.label}” and everything on it?`, tone: "danger", confirmLabel: "Remove" }).result.then((yes) => yes && edit((pr) => ov.removeLane(pr, top.id)));
      } }) });
    host.append(add.node);
    const offDrop = onMediaDrop(".cx-nle-lane-overlay", async (item, lane, e) => {       // a picture dropped on an overlay lane becomes an overlay there
      if (item.kind !== "image" || !p.project) return c.toast.warn({ text: "Only a picture can be laid over the cut." });
      let name = "Image"; try { name = ((await app.api.media()).media || []).find((m) => m.id === item.id).name; } catch { /* keep the generic label */ }
      const row = [...document.querySelectorAll(".cx-nle-lane-overlay")].indexOf(lane), lanes = ov.lanesOf(p.project), laneId = (lanes[lanes.length - 1 - row] || {}).id;       // lanes draw top-first
      const sec = app.timelineView && app.timelineView.timeAt ? app.timelineView.timeAt(e.clientX) : app.playhead.at;
      edit((pr) => ov.addImage(pr, item.id, name, sec, laneId));
    });
    const off = app.on(() => { add.setDisabled && add.setDisabled(!p.project); });
    return () => { off(); offDrop(); const at = app.timelineLanes.indexOf(entry); if (at >= 0) app.timelineLanes.splice(at, 1); add.node.remove(); app.say("timeline.lanes"); };
  },
};
