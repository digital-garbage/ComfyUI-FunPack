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
// nothing. So here a region that does not exist yet is ABSENT. The timeline,
// the picker wheel and the project wizard are not hidden below; they have not
// been built, and adding one is adding a region rather than removing a rule.
// The bin arrived exactly that way.

import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";
import { createTransport } from "./transport.js";
import { createBin } from "./bin.js";

export function build(root, handlers = {}) {
  // Centre: the result, and the prompt under it. The prompt is the one control
  // that is always on screen, because it is the one always being edited.
  const viewer = composer.viewer.media({ empty: "The result of the last run appears here." });

  // Left: what a project is made of, which for now is what it has produced. The
  // bin owns the choice of what the viewer shows -- one place decides, so a
  // click on an older result and a run finishing cannot both be right at once.
  const bin = createBin({ onOpen: (item) => viewer.setSource(item.url, item.kind) });
  const assets = composer.panel.default({
    title: "Assets",
    actions: [bin.control],
    body: bin.host,
  });
  const preview = composer.panel.default({ title: "Preview", body: viewer });
  // The stand-ins below are EMPTY STATES, not labels: they say a region is
  // empty, and settle() takes them down once it is not. A line reading "modules
  // appear here" left above the modules that appeared is a region explaining
  // itself to nobody.
  const promptEmpty = composer.emptyState.default({
    icon: "✎",
    title: "No prompt here",
    hint: "The pipeline decides which of its inputs appear on the main window.",
  });
  const prompt = composer.panel.default({ title: "Prompt", body: promptEmpty });
  // The result gets the room. The prompt is a few lines and a button's worth of
  // controls; at 38% of the height it was mostly empty space taken from the one
  // thing on the page anybody is looking at. Draggable, so anyone writing a long
  // prompt can take it back.
  const centre = composer.splitPane.v({
    size: 74,
    label: "Preview and prompt",
    panes: [preview, prompt],
  });

  // Right: everything about the run. Generation is open, settings folded away
  // -- one is read constantly and the other rarely, and a fold says which is
  // which without needing a second place to look. A settings WINDOW is a
  // separate thing that has not been built; this is not a stand-in for it.
  const generationEmpty = composer.emptyState.default({
    icon: "◎",
    title: "Nothing to set",
    hint: "Modules with settings for a run appear here.",
  });
  const generation = composer.panel.default({ title: "Generation", body: generationEmpty });

  const settingsEmpty = composer.emptyState.default({
    icon: "⚙",
    title: "Nothing to set",
    hint: "Modules with settings that are not about one run appear here.",
  });
  // A host, not a panel: a bordered panel inside the collapsible's own bordered
  // box is two frames drawn around one list of settings.
  const settings = composer.region.stack({ gap: "sm", label: "Settings",
                                           children: [settingsEmpty] });
  // A stack, not a group: a group draws a bordered box with its own padding, so
  // the two panels in this column sat 11px further in than the panel in the
  // left column and inside a second border that meant nothing.
  const properties = composer.region.stack({
    gap: "sm", label: "Properties",
    children: [generation, composer.collapsible.default({ label: "Settings", body: settings })],
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

  // The top bar: the way into everything that is a WINDOW rather than a panel.
  // It holds one button today and will hold the rest as they land -- a place
  // that exists is what stops the next one being wedged into the transport row,
  // which is the one row that has to keep saying what the run is doing.
  const bar = composer.toolbar.default({
    label: "FunPack",
    items: [
      composer.button.sm({
        label: "Models and pipeline",
        onClick: () => { if (handlers.onPipeline) handlers.onPipeline(); },
      }),
    ],
  });

  const page = composer.frame.app({ header: bar, main: workspace, footer: transport });

  root.replaceChildren(page.node);

  // The mount point IS the contract with modules; the region behind it can be
  // rearranged freely as long as the name survives.
  offer("assets.library", assets.body);
  offer("generation.prompt", prompt.body, promptEmpty.node);
  offer("settings.general", settings.node, settingsEmpty.node);
  // One host, five names. Each carries the same stand-in, and settle() takes it
  // down once anything at all has mounted into the panel they share.
  for (const point of ["model", "latent", "sampling", "timing", "post"]) {
    offer(`generation.${point}`, generation.body, generationEmpty.node);
  }

  return { workspace, assets, bin, preview, viewer, prompt, generation, settings,
           properties, transport, bar, page };
}
