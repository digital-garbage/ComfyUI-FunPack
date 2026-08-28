// The app's regions, and the mount points each one offers.
//
// A region declares what it can host. Core never learns these names, and no
// list of them exists anywhere else -- a module asks for one by name, and gets
// it or does not.

import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";

export function build(root) {
  const general = composer.panel.default({
    title: "Settings",
    body: composer.hint.default({ text: "Modules that announced a settings.general mount point." }),
  });
  const generation = composer.panel.default({
    title: "Generation",
    body: composer.hint.default({ text: "Modules that announced a generation.* mount point." }),
  });

  const page = composer.splitPane.h({
    size: 46,
    label: "Settings and generation",
    panes: [general, generation],
  });

  root.replaceChildren(page.node);

  // The mount point IS the contract with modules; the region behind it can be
  // rearranged freely as long as the name survives.
  offer("settings.general", general.body);
  offer("generation.timing", generation.body);
  offer("generation.model", generation.body);

  return { general, generation, page };
}
