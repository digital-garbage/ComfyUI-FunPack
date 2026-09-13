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
//
// A real single-track timeline: clips as wide as they play for, a ruler in
// seconds, and a playhead -- composer's `track.default` does the drawing, this
// file only turns scenes into seconds and seconds back into a selected scene.

import { composer } from "../composer/composer.js";

const EMPTY = {
  icon: "▭",
  title: "Nothing on the timeline",
  hint: "What you generate lands here, in the order it plays.",
};

// How wide a second is, in pixels -- what "zoom" changes. Persisted per
// browser like the media bin's own view choice: a per-viewer convenience, not
// state a run depends on, so nothing breaks if storage is unavailable.
const ZOOM_KEY = "funpack.timeline.zoom";
const ZOOM_LEVELS = { sm: 20, md: 40, lg: 80 };

// No module declares an `fps` role today -- the one pipeline that ships
// (minimax_h3) hands CreateVideo a literal 24.0 rather than a project.video
// input, so there is nothing to read it FROM yet. This mirrors that literal
// rather than inventing a setting nothing produces.
// ponytail: becomes a real per-pipeline value the day a module declares a
// role at project.video.fps, the same way length/target_width/target_height
// already are -- read here with a fallback so that day needs no change here.
const FPS_FALLBACK = 24;

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
  let track = null;
  let pxPerSecond = ZOOM_LEVELS[recallZoom()];
  // The playhead's own position, in seconds -- decoupled from which scene is
  // selected so a click partway through a clip reads as that exact second,
  // not snapped to whichever scene it landed in. Re-synced to the selected
  // scene's start whenever the selection changes from somewhere OTHER than a
  // track click (Add/Remove/Move, a selection made elsewhere, or a different
  // project opened entirely) -- see draw().
  let playhead = 0;
  // What the LAST seek asked for, consumed by the very next draw(). project.
  // select() -- called from onSeek below -- fires its onChange synchronously
  // (boot.js wires it straight to this timeline's own draw()), so a draw can
  // run in the middle of onSeek's own body, before onSeek gets back around to
  // setting the playhead itself. Without this, that reentrant draw reads the
  // OLD `playhead` value against the NEWLY selected scene, and a seek that
  // lands exactly on a clip's boundary -- clamped to the very end of the
  // track, for one -- fails the "inside the span" check and gets silently
  // snapped back to that scene's start. Recording the seek here lets the
  // reentrant draw trust the number just clicked, exactly, instead of
  // re-deriving (and getting wrong) where the playhead belongs.
  let pendingSeek = null;

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
    // Zoom changes seconds->pixels, not the durations behind them -- a bigger
    // track to work with, not a different one.
    trailing: [composer.segmented.sm({
      label: "Zoom",
      value: recallZoom(),
      options: [{ value: "sm", label: "S" }, { value: "md", label: "M" }, { value: "lg", label: "L" }],
      onChange: (level) => {
        rememberZoom(level);
        pxPerSecond = ZOOM_LEVELS[level] || ZOOM_LEVELS.md;
        if (track) track.setZoom(pxPerSecond);
      },
    })],
  });

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

  /** Frames this clip runs for: its own crop, or the project's length. A
   *  negative or NaN length (corrupted project data -- nothing in the UI
   *  produces one) would otherwise pass through as truthy and break the
   *  contiguous, non-decreasing spans the click-seek math in track.js relies
   *  on, the same guard `fps()` below already uses for the same reason. */
  const framesOf = (scene) => (scene.length > 0 ? scene.length : 0) || project.video.length || 1;
  const fps = () => (Number(project.video.fps) > 0 ? Number(project.video.fps) : FPS_FALLBACK);
  const secondsOf = (scene) => framesOf(scene) / fps();

  /** Where each scene starts, in seconds -- contiguous, so a click anywhere
   *  on the track belongs to exactly one of them. */
  function spans() {
    let at = 0;
    return project.scenes.map((scene) => {
      const start = at;
      const duration = secondsOf(scene);
      at += duration;
      return { scene, start, duration };
    });
  }

  function items() {
    return project.scenes.map((scene, i) => {
      const { start, duration } = spans()[i];
      return {
        id: scene.id,
        // The number is what a scene is called before it has any text, and the
        // text is what it is called after -- a strip nobody can read their own
        // project off.
        label: scene.text ? `${i + 1}. ${scene.text}` : `Scene ${i + 1}`,
        badge: String(i + 1),
        thumb: scene.result || null,
        icon: "▦",
        start, duration,
        rating: scene.rating || null,
        excluded: Boolean(scene.excluded),
      };
    });
  }

  function draw() {
    const scenes = project.scenes;
    if (!scenes.length) {
      host.set([empty]);
      track = null;
      return;
    }
    const built = items();
    const current = built.find((i) => i.id === project.selectedId);
    if (current) {
      if (pendingSeek && pendingSeek.id === current.id) {
        // Trust the seek just clicked exactly, rather than re-deriving it --
        // this draw may be running INSIDE onSeek's own body (project.select's
        // onChange fires synchronously), before onSeek gets back around to
        // setting the playhead itself, and a seek clamped to the very edge of
        // the track would otherwise fail a span check and snap back to the
        // scene's start.
        playhead = pendingSeek.seconds;
      } else {
        // Anything else that changed the selection -- Add, Remove, Move, a
        // pick made elsewhere, a different project opened entirely -- only
        // moved the selection; this is what pulls the playhead back onto it.
        // Always, not just when the old value falls outside the new scene's
        // span: two scenes from two different projects can share a span by
        // pure coincidence (even the same id, if one is ever reused), and
        // trusting "still numerically inside" as a proxy for "still the same
        // seek" is exactly what lets a stale playhead survive a project it
        // has nothing to do with.
        playhead = current.start;
      }
    }
    pendingSeek = null;
    if (!track) {
      track = composer.track.default({
        label: "Timeline",
        items: built,
        selection: project.selectedId ? [project.selectedId] : [],
        playhead,
        pxPerSecond,
        onSeek: (seconds, item) => {
          playhead = seconds;
          if (item) pendingSeek = { id: item.id, seconds };
          // select() no-ops, without announcing, when the click landed
          // inside the scene already current -- the common case for a pure
          // seek, and when it does no reentrant draw() happens at all. The
          // playhead still needs to move, so it is set directly rather than
          // only through whatever redraw a real selection change would have
          // triggered. Cleared right after, unconditionally: a no-op select
          // leaves no reentrant draw to consume pendingSeek, and without this
          // it can survive to be wrongly replayed by a LATER, unrelated draw
          // whose current scene happens to share this one's id.
          if (item) project.select(item.id);
          pendingSeek = null;
          announce();
          track.setPlayhead(playhead);
          track.setValue(project.selectedId ? [project.selectedId] : []);
        },
        // Ids, resolved to a position HERE, now -- not carried from whenever
        // the drag started. A remove or an undo can redraw the track while a
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
      host.set([controls, track]);
      return;
    }
    track.setItems(built);
    track.setValue(project.selectedId ? [project.selectedId] : []);
    track.setPlayhead(playhead);
  }

  draw();

  return {
    node: host.node,
    draw,
    destroy() {
      if (track) track.destroy();
      controls.destroy();
      empty.destroy();
      host.destroy();
    },
  };
}
