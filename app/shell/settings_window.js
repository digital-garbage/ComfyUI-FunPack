// One home for everything that is a PREFERENCE about the app or the pipeline,
// rather than an edit to the project on screen -- replacing five separate
// menu items (Models and pipeline, Updates, Node packs, Log, Temp files) that
// nothing tied together and nothing let you search.
//
// A section is either MOUNTED here (About, today) or a DEEP LINK: picking it
// closes this window and opens the real one. Every deep-linked window here
// (pipeline_window, updates, packs, log, tempfiles) already owns real,
// tested modal chrome of its own -- rebuilding each into a mountable pane is
// real surgery on working code for a visual nicety, not a fix for anything
// broken. This gets the one thing that mattered (one findable, searchable
// place to start) without touching any of that.

import { composer } from "../composer/composer.js";

let current = null;

/**
 * createSettingsWindow({ sections, gitStatus }) -> { open, close }
 *
 * `sections`: [{ id, title, subtitle, keywords, icon,
 *                mount() -> handle }  -- OR --  { action() }].
 * A section with `action` is a deep link: picking it closes this window and
 * calls `action`. A section with `mount` is drawn in place; `mount()` returns
 * a composer element handle (its own `destroy()` is the cleanup, called when
 * another section is picked or the window closes).
 *
 * `gitStatus` is injected, not imported, for the same reason every other
 * shell window takes its network calls as props: testable without a server,
 * and unable to reach for one that was not handed to it.
 */
export function createSettingsWindow({ sections = [], gitStatus } = {}) {
  const all = [aboutSection(gitStatus), ...sections];

  function open(id) {
    if (current) { current.show(id); return current; }

    let activeCleanup = null;
    const contentStack = composer.region.stack({ gap: "sm", fill: true });

    function showSection(pick) {
      const spec = all.find((s) => s.id === pick) || all[0];
      if (!spec) return;
      if (activeCleanup) { activeCleanup.destroy?.(); activeCleanup = null; }
      if (spec.action) {
        // Deep link: this window is not what is being asked for, only how
        // it was found. Close first -- two full-size modals stacked is not
        // "unified", it is one more window in front of the one just opened.
        close();
        spec.action();
        return;
      }
      const content = spec.mount();
      activeCleanup = content;
      contentStack.set([
        composer.header.md({ text: spec.title }),
        spec.subtitle ? composer.hint.default({ text: spec.subtitle }) : null,
        content,
      ]);
      nav.setValue(spec.id);
    }

    const nav = composer.filterList.md({
      items: all.map((s) => ({ id: s.id, label: s.title, hint: s.subtitle, icon: s.icon })),
      value: id,
      onChange: showSection,
      placeholder: "Search settings",
    });

    const body = composer.splitPane.h({ panes: [nav, contentStack], size: 25, min: 15, label: "Settings" });
    const modal = composer.modal.generic({
      title: "Settings", size: "lg", body,
      onClose: () => {
        if (activeCleanup) { activeCleanup.destroy?.(); activeCleanup = null; }
        if (current === handle) current = null;
      },
    });

    const handle = { node: modal.node, close: () => modal.close(), show: showSection };
    current = handle;
    showSection(id || (all[0] && all[0].id));
    return handle;
  }

  function close() {
    if (current) current.close();
  }

  return { open, close };
}

/** Version and branch -- the one thing everyone with the window open at all
 *  might be there to check, so it is built in rather than one more click away. */
function aboutSection(gitStatus) {
  return {
    id: "about", title: "About FunPack",
    subtitle: "The version and branch running right now.",
    keywords: "about version commit branch",
    icon: "◉",
    mount() {
      const stack = composer.region.stack({ gap: "sm" });
      const rows = composer.region.stack({ gap: "none" });
      const row = (label, value) => composer.settingsRow.default({
        label, control: composer.text.sm({ text: value }) });
      rows.set([row("Version", "…"), row("Branch", "…")]);
      stack.set([rows]);
      if (gitStatus) {
        gitStatus().then((s) => {
          rows.set([row("Version", s.version || "unknown"), row("Branch", s.branch || "?")]);
        }).catch(() => {
          rows.set([composer.hint.default({ text: "Could not read version information." })]);
        });
      }
      return stack;
    },
  };
}
