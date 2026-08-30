// The app's regions, and the mount points each one offers.
//
// A region declares what it can host. Core never learns these names, and no
// list of them exists anywhere else -- a module asks for one by name, and gets
// it or does not.
//
// The shape is the one v4 called Simple mode, built the other way round. v4's
// Simple mode was the Editor with about ten `display: none !important` rules
// over it, which is not a smaller app but the same app wearing a mask -- and it
// broke: a collapsed column was marked [hidden], and nothing can slide a
// display:none element into view, so a panel button did nothing at all and said
// nothing. So here a region that does not exist yet is ABSENT. The timeline, the
// media bin's view modes, the pipeline window and the wheel are not hidden
// below; they have not been built, and adding one is adding a region rather
// than removing a rule.

import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";
import { createTransport } from "./transport.js";

export function build(root, handlers = {}) {
  // Left: what a project is made of. Empty until the media bin exists, and an
  // empty state rather than an absent panel, because "there is nothing here
  // yet" and "this part of the app is missing" must not look the same.
  const assets = composer.panel.default({
    title: "Assets",
    body: composer.emptyState.default({
      icon: "▤",
      title: "Nothing in the bin",
      hint: "Media you add to the project appears here.",
    }),
  });

  // Centre: the result, and the prompt under it. The prompt is the one control
  // that is always on screen, because it is the one always being edited.
  const viewer = composer.viewer.media({ empty: "The result of the last run appears here." });
  const preview = composer.panel.default({ title: "Preview", body: viewer });
  const prompt = composer.panel.default({
    title: "Prompt",
    body: composer.hint.default({ text: "Modules that announced generation.prompt." }),
  });
  const centre = composer.splitPane.v({
    size: 62,
    label: "Preview and prompt",
    panes: [preview, prompt],
  });

  // Right: everything about the run. Generation is open, settings folded away
  // -- one is read constantly and the other rarely, and a fold says which is
  // which without needing a second place to look. A settings WINDOW is a
  // separate thing that has not been built; this is not a stand-in for it.
  const generation = composer.panel.default({
    title: "Generation",
    body: composer.hint.default({ text: "Modules that announced a generation.* mount point." }),
  });
  const settings = composer.panel.default({
    body: composer.hint.default({ text: "Modules that announced settings.general." }),
  });
  const properties = composer.group.default({
    rows: [generation, composer.collapsible.default({ label: "Settings", body: settings })],
  });

  const workspace = composer.workspace.docked({
    id: "main",
    leftLabel: "Assets",
    rightLabel: "Properties",
    left: assets,
    centre,
    right: properties,
  });

  // The transport. One row, always visible, and the place a run is started and
  // reported -- an inert setting is said HERE, next to Generate, not in the
  // panel that would have enabled it.
  const transport = createTransport(handlers);

  const page = composer.frame.app({ main: workspace, footer: transport });

  root.replaceChildren(page.node);

  // The mount point IS the contract with modules; the region behind it can be
  // rearranged freely as long as the name survives.
  offer("assets.library", assets.body);
  offer("generation.prompt", prompt.body);
  offer("settings.general", settings.body);
  offer("generation.model", generation.body);
  offer("generation.latent", generation.body);
  offer("generation.sampling", generation.body);
  offer("generation.timing", generation.body);
  offer("generation.post", generation.body);

  return { workspace, assets, preview, viewer, prompt, generation, settings,
           properties, transport, page };
}
