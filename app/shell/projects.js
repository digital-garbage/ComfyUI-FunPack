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
export function createProject({ onChange, onError } = {}) {
  let project = null;
  let selected = null;
  let timer = null;
  let saving = null;      // the PUT in flight, so a queued one waits for it
  let dirty = false;

  const changed = () => { if (onChange) onChange(); };

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
    async start() {
      const found = await list();
      project = found.length ? await read(found[0].id) : await create("Untitled");
      selected = (project.scenes || [])[0]?.id ?? null;
      changed();
      return project;
    },

    async open(id) {
      await flush();
      project = await read(id);
      selected = (project.scenes || [])[0]?.id ?? null;
      changed();
      return project;
    },

    async newProject(name) {
      await flush();
      project = await create(name);
      selected = (project.scenes || [])[0]?.id ?? null;
      changed();
      return project;
    },

    select(id) {
      if (!sceneAt(id) || id === selected) return;
      selected = id;
      changed();                          // selection is not saved: it is a view
    },

    /** The text of one scene. The prompt box on the main window edits this. */
    setText(id, text) {
      const scene = sceneAt(id);
      if (!scene || scene.text === text) return;
      scene.text = text;
      scheduleSave();
      changed();
    },

    addScene() {
      if (!project) return null;
      // A new scene lands AFTER the selected one, not at the end: adding partway
      // through a timeline is how a scene gets inserted, and appending silently
      // would put it somewhere the user was not looking.
      const at = project.scenes.findIndex((s) => s.id === selected);
      const scene = { id: newId(), text: "", result: null };
      project.scenes.splice(at < 0 ? project.scenes.length : at + 1, 0, scene);
      selected = scene.id;                // you add a scene in order to fill it
      scheduleSave();
      changed();
      return scene;
    },

    removeScene(id) {
      if (!project || !sceneAt(id)) return;
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
      const [scene] = project.scenes.splice(at, 1);
      project.scenes.splice(to, 0, scene);
      scheduleSave();
      changed();
    },

    /** Attach what a run produced to the scene it was started from. */
    setResult(id, result) {
      const scene = sceneAt(id);
      if (!scene) return;
      scene.result = result;
      scheduleSave();
      changed();
    },

    flush,
    get unsaved() { return dirty; },
  };
}
