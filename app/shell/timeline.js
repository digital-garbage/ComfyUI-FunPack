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
    }));
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
      });
      host.set([controls, strip]);
      return;
    }
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
      controls.destroy();
      empty.destroy();
      host.destroy();
    },
  };
}
