// The project: an ordered list of scenes, and the only thing in the app that
// has to survive a reload.
//
// Held here rather than in the timeline that draws it, for the reason the
// pipeline is held in boot: one store of live values. A timeline that owned the
// project would make "what is on screen" and "what a run uses" two questions
// with two answers.
//
// Saving is DEBOUNCED and last-write-wins per project. Typing a prompt is one
// edit per keystroke and each is a PUT of the whole project; without this the
// server answers them out of order and an older body lands last.

const BASE = "/funpack/api/projects";
const SAVE_AFTER = 600;

/**
 * A scene id the server will keep.
 *
 * Twelve hex characters is what core/projects.py generates and the only shape it
 * accepts back; anything else it replaces with one of its own. Minting a valid
 * one here means a new scene is addressable the moment it appears, instead of
 * being unselectable until a save round trip has answered.
 */
function newId() {
  const bytes = new Uint8Array(6);
  (globalThis.crypto || {}).getRandomValues?.(bytes);
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function json(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = "";
    try { detail = ((await res.json()).problems || [])[0] || ""; } catch { /* not JSON */ }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.status === 204 ? null : res.json();
}

export const list = () => json("GET", BASE).then((d) => d.projects || []);
export const create = (name) => json("POST", BASE, { name });
export const read = (id) => json("GET", `${BASE}/${encodeURIComponent(id)}`);
export const remove = (id) => json("DELETE", `${BASE}/${encodeURIComponent(id)}`);
const put = (project) => json("PUT", `${BASE}/${encodeURIComponent(project.id)}`, project);

/**
 * createProject({ onChange, onError }) -> the live project.
 *
 * `onChange` fires for anything that alters what is on screen, including the
 * selection. Nothing here draws; the timeline subscribes.
 */
export function createProject({ onChange, onError, onOpen } = {}) {
  let project = null;
  let selected = null;
  // Every project this browser knows about, for the File menu. Read once at
  // startup and kept in step here rather than re-fetched each time the menu
  // opens: a listing is not a thing that changes behind the user's back.
  let recent = [];
  // Undo, the cheap way: a project is plain JSON and a few kilobytes, so a
  // snapshot before each edit costs nothing worth measuring and needs no
  // per-action inverse. Bounded, or a long session keeps every version of a
  // project it has ever had.
  // ponytail: whole-project snapshots, capped at 40. Per-field diffs only if a
  // project ever gets big enough for the copy to show up in a profile.
  const LIMIT = 40;
  let past = [];
  let future = [];
  // The selection travels with the snapshot. It is view state and not saved,
  // but undoing a removal and landing on a different scene than the one that
  // came back is not what anyone means by "take that back".
  const copy = () => ({ project: JSON.parse(JSON.stringify(project)), selected });
  let timer = null;
  let saving = null;      // the PUT in flight, so a queued one waits for it
  let dirty = false;

  const changed = () => { if (onChange) onChange(); };

  /** Before an edit. Anything that changes what is SAVED calls this; selecting
   *  a scene does not, because a view is not an edit. */
  function remember() {
    if (!project) return;
    past.push(copy());
    if (past.length > LIMIT) past.shift();
    // A new edit ends the redo line. Keeping it would let a user redo their way
    // into a version that never followed from what they are looking at.
    future = [];
  }

  function step(from, to) {
    if (!project || !from.length) return false;
    to.push(copy());
    const was = from.pop();
    project = was.project;
    selected = was.selected;
    // The scene that was selected may be gone -- stepping FORWARD into a state
    // where it had been removed. Fall back rather than leaving a selection that
    // names nothing, which reads as "no scene" in every panel that draws it.
    if (!(project.scenes || []).some((sc) => sc.id === selected)) {
      selected = (project.scenes || [])[0]?.id ?? null;
    }
    scheduleSave();
    opened();                             // the project was REPLACED, not edited
    return true;
  }
  // A DIFFERENT project is open now -- which is not the same event as a field
  // changing, and the things that follow a project are not the things that
  // follow an edit. Fired by every route in: start, open, new.
  const opened = () => { changed(); if (onOpen) onOpen(project); };
  const forget = () => { past = []; future = []; };

  function scheduleSave() {
    dirty = true;
    if (timer) clearTimeout(timer);
    timer = setTimeout(flush, SAVE_AFTER);
  }

  async function flush() {
    if (timer) { clearTimeout(timer); timer = null; }
    if (!project || !dirty) return;
    // One PUT at a time. Two overlapping writes of a whole project can land in
    // either order, and the loser is a version of the project the user has
    // already moved past.
    if (saving) { await saving.catch(() => {}); if (!dirty) return; }
    dirty = false;
    const body = { ...project };
    saving = put(body).then((saved) => {
      // Only the fields the server owns are taken back: replacing the whole
      // project would overwrite whatever was typed while the PUT was in flight.
      if (project && saved && project.id === saved.id) project.updated_at = saved.updated_at;
    }).catch((err) => {
      dirty = true;                       // it is still unsaved, so say so
      if (onError) onError(err);
    }).finally(() => { saving = null; });
    return saving;
  }

  const sceneAt = (id) => (project ? (project.scenes || []).find((s) => s.id === id) : null);

  return {
    get project() { return project; },
    get scenes() { return project ? project.scenes || [] : []; },
    get selectedId() { return selected; },
    get selected() { return sceneAt(selected); },

    /** The most recent project, or a new one when there are none. */
    get recent() { return recent; },

    async start() {
      const found = await list();
      recent = found;
      project = found.length ? await read(found[0].id) : await create("Untitled");
      if (!found.length) recent = [{ id: project.id, name: project.name }];
      selected = (project.scenes || [])[0]?.id ?? null;
      forget();                           // a project just opened has no past
      opened();
      return project;
    },

    async open(id) {
      await flush();
      project = await read(id);
      selected = (project.scenes || [])[0]?.id ?? null;
      forget();                           // a project just opened has no past
      opened();
      return project;
    },

    async newProject(name) {
      await flush();
      project = await create(name);
      recent = [{ id: project.id, name: project.name }, ...recent];
      selected = (project.scenes || [])[0]?.id ?? null;
      forget();                           // a project just opened has no past
      opened();
      return project;
    },

    select(id) {
      if (!sceneAt(id) || id === selected) return;
      selected = id;
      changed();                          // selection is not saved: it is a view
    },

    /** What it is called. The listing in the File menu reads this too, so both
     *  move together rather than the menu going stale until a reload. */
    rename(name) {
      const clean = String(name || "").trim();
      if (!project || !clean || project.name === clean) return;
      remember();
      project.name = clean;
      const listed = recent.find((p) => p.id === project.id);
      if (listed) listed.name = clean;
      scheduleSave();
      changed();
    },

    /** What the whole project generates at, whichever scene is current. */
    get video() { return (project && project.video) || {}; },

    setVideo(key, value) {
      if (!project) return;
      project.video = project.video || {};
      if (project.video[key] === value) return;
      remember();
      project.video[key] = value;
      scheduleSave();
      // The timeline's clip widths and its ruler are computed from the project's
      // length. This said "nothing on screen draws from this" and was true when
      // it was written -- the timeline started reading it an hour later, and a
      // comment is not a thing the next change has to keep true.
      changed();
    },

    /** The text of one scene. The prompt box on the main window edits this. */
    setText(id, text) { this.setScene(id, "text", text); },

    addScene() {
      if (!project) return null;
      // A new scene lands AFTER the selected one, not at the end: adding partway
      // through a timeline is how a scene gets inserted, and appending silently
      // would put it somewhere the user was not looking.
      remember();
      const at = project.scenes.findIndex((s) => s.id === selected);
      const scene = { id: newId(), text: "", result: null, length: null, rating: null };
      project.scenes.splice(at < 0 ? project.scenes.length : at + 1, 0, scene);
      selected = scene.id;                // you add a scene in order to fill it
      scheduleSave();
      changed();
      return scene;
    },

    removeScene(id) {
      if (!project || !sceneAt(id)) return;
      remember();
      const at = project.scenes.findIndex((s) => s.id === id);
      project.scenes.splice(at, 1);
      if (selected === id) {
        const next = project.scenes[at] || project.scenes[at - 1];
        selected = next ? next.id : null;
      }
      scheduleSave();
      changed();
    },

    move(id, by) {
      if (!project) return;
      const at = project.scenes.findIndex((s) => s.id === id);
      const to = at + by;
      if (at < 0 || to < 0 || to >= project.scenes.length) return;
      remember();
      const [scene] = project.scenes.splice(at, 1);
      project.scenes.splice(to, 0, scene);
      scheduleSave();
      changed();
    },

    /** One field of one scene. Everything a scene holds but its id goes through
     *  here, so there is one place that saves and one that says it changed. */
    setScene(id, key, value) {
      const scene = sceneAt(id);
      if (!scene || scene[key] === value) return;
      remember();
      scene[key] = value;
      scheduleSave();
      changed();
    },

    /** Attach what a run produced to the scene it was started from. */
    setResult(id, result) { this.setScene(id, "result", result); },

    /**
     * The same, but for a run that may have finished after the user switched
     * to a DIFFERENT project. A run takes minutes and nothing stops someone
     * from opening another project while one is in flight -- `setResult`
     * alone would look the scene id up in whatever is open NOW, find nothing,
     * and drop the result with no trace: the GPU time spent, the image gone.
     */
    async setResultFor(projectId, sceneId, result) {
      if (project && project.id === projectId) {
        this.setResult(sceneId, result);
        return;
      }
      try {
        const doc = await read(projectId);
        const scene = (doc.scenes || []).find((s) => s.id === sceneId);
        if (!scene) return; // the scene itself is gone -- nothing to attach this to
        scene.result = result;
        await put(doc);
      } catch (err) {
        if (onError) onError(err);
      }
    },

    /** Step back, or forward. True when something moved. */
    undo() { return step(past, future); },
    redo() { return step(future, past); },
    get canUndo() { return Boolean(project) && past.length > 0; },
    get canRedo() { return Boolean(project) && future.length > 0; },

    flush,
    get unsaved() { return dirty; },
  };
}
