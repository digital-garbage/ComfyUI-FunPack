// The Properties zone: Project and Scene tabs, titled for what they show.
import { composer as c } from "../../composer/composer.js";
import { effFrames, effFps, genUnitId } from "../../shell/scenes.js";
import { sceneRows } from "./scene.js";
import { bin } from "../../shell/bin.js";
import { projectRows } from "./project.js";

export default {
  id: "inspector",
  mount: "inspector",
  needs: ["project"],
  setup({ host, app }) {
    const p = app.project;
    let tab = "scene", key = "";
    const body = c.region.stack({ gap: "sm", label: "Properties" });
    const tabs = c.segmented.sm({ label: "Properties", options: [{ value: "project", label: "Project" }, { value: "scene", label: "Scene" }], value: tab,
      onChange: (v) => { tab = v; draw(true); } });
    host.append(tabs.node, body.node);

    const titled = () => {
      const sc = p.selected;
      if (tab === "project" || !sc) return app.title && app.title("inspector", tab === "project" ? "Project" : "Scene");
      const unit = p.scenes.filter((s) => genUnitId(s) === genUnitId(sc));
      app.title && app.title("inspector", `Scene · ${p.scenes.indexOf(sc) + 1}${unit.length > 1 ? ` · cut ${unit.indexOf(sc) + 1}/${unit.length}` : ""}`);
    };
    const sections = app.sceneSections || [];       // extra rows other features put under the scene's own: { rows(scene, project), key(scene, project) }
    // The scene's text is in the key, so an edit made elsewhere (the Story box) is shown -- but not while this panel's
    // own box is being typed in: that would rebuild it under the caret. It catches up when the focus leaves.
    let shownText;
    const typing = () => body.node.contains(document.activeElement) && /^(input|textarea)$/i.test(document.activeElement.tagName);
    // Catch up once focus leaves the panel. Focus moving to a button or select inside it is a click under way: rebuilding
    // between its mouse-down and mouse-up dropped the click; the change it makes redraws afterwards anyway.
    body.node.addEventListener("focusout", (e) => { if (!(e.relatedTarget && body.node.contains(e.relatedTarget))) setTimeout(() => { if (!typing()) draw(); }); });
    function draw(force) {      // not on every keystroke: that would rebuild the box being typed in
      const sc = p.selected, open = p.project;
      const next = `${tab}|${open && open.id}|${p.selectedId}|${tab === "project" && open ? [open.num_frames_per_scene, open.frame_rate, JSON.stringify(open.video || {}), open.postfix_enabled !== false, open.generation_mode, open.export_size_from, open.width, open.height, open.scenes.map((s) => s.frames_mode === "custom"), Object.keys(open.scene_renders || {}), open.scenes.length] : ""}|${tab === "scene" && sc && open ? [effFrames(sc, open), effFps(sc, open), p.scenes.indexOf(sc), sc.frames_mode, sc.fps_mode, p.scenes.length, sc.source_image, (sc.references || []).join(), sc.source_in, sc.source_dur, sc.removed_from_plan, (sc.source || {}).type, sc.excluded, typing() ? shownText : (shownText = sc.text), ((open.scene_renders || {})[sc.id] || {}).durationSec, Boolean(((open.scene_renders || {})[sc.id] || {}).media), bin.version, ...sections.map((s) => { try { return s.key(sc, open); } catch { return ""; } })] : ""}`;
      titled();
      if (!force && next === key) return;
      key = next;
      body.set((tab === "scene" ? [...sceneRows(p, app), ...(p.selected ? sections.flatMap((s) => { try { return s.rows(p.selected, p); } catch { return []; } }) : [])] : projectRows(p, app.pipeline && app.pipeline.slots(), app.api))); 
    }
    draw(true);
    const offSlots = app.pipeline && app.pipeline.subscribe ? app.pipeline.subscribe(() => draw(true)) : null;     // the pipeline arrives after the project
    const offApp = app.on((what) => draw(what === "open"));
    return () => { offApp(); if (offSlots) offSlots(); };
  },
};
