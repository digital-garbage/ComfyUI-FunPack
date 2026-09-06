// The timeline: the project's scenes, in the order they play.
//
// It draws and it dispatches. The project itself lives in projects.js, so this
// file holds no scene state of its own -- a timeline that kept its own copy is
// how "what is on screen" and "what a run uses" become two questions.
//
// One scene is always current, and the prompt box on the main window edits THAT
// scene. v4 had one prompt for the whole project and split it by markers in the
// text, which meant the text was both the content and the structure: editing a
// scene could re-cut every other one.

import { composer } from "../composer/composer.js";

const EMPTY = {
  icon: "▭",
  title: "Nothing on the timeline",
  hint: "What you generate lands here, in the order it plays.",
};

// How wide a clip is at rest, in pixels -- what "zoom" changes. Persisted per
// browser like the media bin's own view choice: a per-viewer convenience, not
// state a run depends on, so nothing breaks if storage is unavailable.
const ZOOM_KEY = "funpack.timeline.zoom";
const ZOOM_LEVELS = { sm: 48, md: 72, lg: 128 };

function recallZoom() {
  try { return ZOOM_LEVELS[window.localStorage.getItem(ZOOM_KEY)] ? window.localStorage.getItem(ZOOM_KEY) : "md"; }
  catch { return "md"; }
}
function rememberZoom(level) {
  try { window.localStorage.setItem(ZOOM_KEY, level); } catch { /* private mode */ }
}

/**
 * createTimeline({ project, onSelect }) -> { node, draw, destroy }
 *
 * `project` is the store from projects.js. `onSelect` fires whenever the current
 * scene changes, including when a removal moves it -- the prompt box follows it.
 */
export function createTimeline({ project, onSelect } = {}) {
  const host = composer.region.stack({ gap: "sm", fill: true });
  const empty = composer.emptyState.default(EMPTY);
  let strip = null;

  // Rebuilt on every draw, so a stale handler cannot act on a scene that has
  // moved. The buttons that act on "the current scene" read it at click time
  // rather than closing over it for the same reason.
  const controls = composer.toolbar.default({
    label: "Scenes",
    items: [
      composer.button.sm({ label: "Add scene", onClick: () => add() }),
      composer.button.sm({ label: "Remove", onClick: () => remove() }),
      composer.iconButton.sm({ icon: "◀", label: "Move earlier", onClick: () => move(-1) }),
      composer.iconButton.sm({ icon: "▶", label: "Move later", onClick: () => move(1) }),
    ],
    // Zoom changes clip WIDTH, not the frame counts behind them -- a bigger
    // strip to work with, not a different one.
    trailing: [composer.segmented.sm({
      label: "Zoom",
      value: recallZoom(),
      options: [{ value: "sm", label: "S" }, { value: "md", label: "M" }, { value: "lg", label: "L" }],
      onChange: (level) => { rememberZoom(level); applyZoom(level); },
    })],
  });

  function applyZoom(level) {
    if (strip) strip.node.style.setProperty("--strip-w", `${ZOOM_LEVELS[level] || ZOOM_LEVELS.md}px`);
  }

  function add() {
    project.addScene();
    announce();
  }

  function remove() {
    const id = project.selectedId;
    // The last scene stays. A project with no scenes has nowhere to type, and
    // the Add button would be the only thing on the timeline that did anything.
    if (!id || project.scenes.length <= 1) return;
    project.removeScene(id);
    announce();
  }

  function move(by) {
    if (project.selectedId) project.move(project.selectedId, by);
  }

  function announce() {
    if (onSelect) onSelect(project.selected || null);
  }

  /** Frames this clip runs for: its own crop, or the project's length. */
  const lengthOf = (scene) => scene.length || project.video.length || 1;

  function items() {
    return project.scenes.map((scene, i) => ({
      id: scene.id,
      // The number is what a scene is called before it has any text, and the
      // text is what it is called after -- a strip of "Scene 1..8" is a strip
      // nobody can read their own project off.
      label: scene.text ? `${i + 1}. ${scene.text}` : `Scene ${i + 1}`,
      badge: String(i + 1),
      thumb: scene.result || null,
      icon: "▦",
      // A clip as wide as it is long, so the strip reads as time rather than as
      // a row of equal boxes. Bounded: one very long scene beside several short
      // ones must not squeeze the rest to nothing.
      weight: lengthOf(scene),
      rating: scene.rating || null,
      excluded: Boolean(scene.excluded),
    }));
  }

  /** Where each scene starts, in frames. The ruler is drawn from this. */
  function marks() {
    let at = 0;
    return project.scenes.map((scene) => {
      const start = at;
      at += lengthOf(scene);
      return { id: scene.id, start, length: lengthOf(scene) };
    });
  }

  // The ruler. Its ticks are the scene boundaries rather than a fixed interval:
  // what a person looks for on this timeline is where one scene becomes the
  // next, and at 24fps a tick per second is a picket fence.
  const ruler = composer.ruler.default({ label: "Scenes" });

  function drawRuler() {
    const spans = marks();
    const total = spans.reduce((sum, m) => sum + m.length, 0) || 1;
    ruler.set(spans.map((m, i) => ({
      at: m.start / total,
      label: `${i + 1}`,
      hint: `${m.start}`,
    })), total);
  }

  function draw() {
    const scenes = project.scenes;
    if (!scenes.length) {
      host.set([empty]);
      strip = null;
      return;
    }
    if (!strip) {
      strip = composer.gallery.strip({
        label: "Scenes",
        items: items(),
        selection: project.selectedId ? [project.selectedId] : [],
        onActivate: (item) => { project.select(item.id); announce(); },
        // Ids, resolved to a position HERE, now -- not carried from whenever
        // the drag started. A remove or an undo can redraw the strip while a
        // drag is still in flight (the mouse held down is a different input
        // channel from the keyboard shortcut that triggers one), and a
        // position captured at dragstart would then name whatever has since
        // taken that slot, not what was actually picked up.
        onReorder: (draggedId, targetId) => {
          const from = project.scenes.findIndex((s) => s.id === draggedId);
          const to = project.scenes.findIndex((s) => s.id === targetId);
          if (from < 0 || to < 0 || from === to) return;
          project.move(draggedId, to - from);
          announce();
        },
      });
      host.set([controls, ruler, strip]);
      applyZoom(recallZoom());
      drawRuler();
      return;
    }
    drawRuler();
    // setItems then setValue: setValue redraws against whatever items are
    // current, so seeding the selection first marks a row that is about to be
    // replaced and the strip comes back with nothing on.
    strip.setItems(items());
    strip.setValue(project.selectedId ? [project.selectedId] : []);
  }

  draw();

  return {
    node: host.node,
    draw,
    destroy() {
      if (strip) strip.destroy();
      ruler.destroy();
      controls.destroy();
      empty.destroy();
      host.destroy();
    },
  };
}
