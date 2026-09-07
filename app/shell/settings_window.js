// One home for everything that is a PREFERENCE about the app or the pipeline,
// rather than an edit to the project on screen -- replacing five separate
// menu items (Models and pipeline, Updates, Node packs, Log, Temp files) that
// nothing tied together and nothing let you search.
//
// Every section is MOUNTED here, in place, beside the same always-visible
// nav list -- picking "Node packs" does not leave this window any more than
// clicking a different System Settings pane leaves System Settings. Each of
// pipeline_window.js/updates.js/packs.js/logwindow.js/tempfiles.js already
// had its content built separately from its own modal chrome (a `body`
// region.stack, footer buttons set on the modal handle); their `mount()`
// exports are that same content, taking `setFooter`/`close` from whoever is
// hosting them instead of owning a modal directly. Their `open()` exports
// are unchanged for anyone opening one standalone.

import { composer } from "../composer/composer.js";

/**
 * createSettingsWindow({ sections, gitStatus, systemInfo }) -> { open, close }
 *
 * `sections`: [{ id, title, subtitle, keywords, icon,
 *                mount({ setFooter, close }) -> handle }].
 * `mount()` returns a composer element handle (its own `destroy()` is the
 * cleanup, called when another section is picked or the window closes).
 * `setFooter`/`close` act on this window's own modal chrome -- a section
 * does not know or care whether it is standing alone or hosted here.
 *
 * `gitStatus`/`systemInfo` are injected, not imported, for the same reason
 * every other shell window takes its network calls as props: testable
 * without a server, and unable to reach for one that was not handed to it.
 *
 * `current` lives HERE, per call, not at module scope: a module-level
 * singleton would let a second independent settings window (there is only
 * ever one in this app today, but nothing enforced that) silently redraw
 * the FIRST one's modal instead of opening its own.
 */
export function createSettingsWindow({ sections = [], gitStatus, systemInfo } = {}) {
  const all = [aboutSection(gitStatus, systemInfo), ...sections];
  let current = null;

  function open(id) {
    if (current) { current.show(id); return current; }

    let activeCleanup = null;
    const contentStack = composer.region.stack({ gap: "sm", fill: true });

    function showSection(pick) {
      const spec = all.find((s) => s.id === pick) || all[0];
      if (!spec) return;
      if (activeCleanup) { activeCleanup.destroy?.(); activeCleanup = null; }
      // Cleared before mounting: a section that sets no footer of its own
      // (About) must not go on showing the PREVIOUS section's buttons.
      modal.setFooter({});
      const content = spec.mount({
        setFooter: (f) => modal.setFooter(f),
        close: () => modal.close("done"),
      });
      activeCleanup = content;
      contentStack.set([
        composer.header.md({ text: spec.title }),
        spec.subtitle ? composer.hint.default({ text: spec.subtitle }) : null,
        content,
      ]);
      nav.setValue(spec.id);
    }

    const nav = composer.filterList.md({
      // No hint line here -- an icon and a name, the way v4's own sidebar
      // reads. What a section IS goes in its own head once picked, not
      // repeated under every row whether picked or not.
      items: all.map((s) => ({
        id: s.id, label: s.title, icon: s.icon, keywords: s.keywords,
        tone: s.tone || "neutral",
      })),
      value: id,
      onChange: showSection,
      placeholder: "Search settings",
    });
    // A whole pane, not the compact dropdown-style list this element is
    // usually a small part of.
    nav.node.classList.add("cx-filter-full");
    contentStack.node.classList.add("cx-settings-content");

    // A retractable rail, not a resizable split: v4's own sidebar is a 52px
    // icon strip that expands to a full list on hover/focus and OVERLAYS the
    // content rather than pushing it over -- .cx-settings-body/.cx-settings-rail
    // do the positioning; splitPane's drag handle would be the wrong affordance
    // for something that is not meant to stay open.
    const body = composer.region.stack({ gap: "none", fill: true, children: [nav, contentStack] });
    body.node.classList.add("cx-settings-body");
    nav.node.classList.add("cx-settings-rail");
    const modal = composer.modal.generic({
      title: "Settings", size: "xl", body,
      // Mirrors pipeline_window.js's own standalone modal: a half-finished
      // edit (Models and pipeline's draft, most concretely) must not vanish
      // because a click landed on the backdrop. This window has no way to
      // ask "does the section on screen have unsaved work" per section, so
      // it holds the door for all of them, the same conservative call
      // pipeline_window.js already made for itself -- the X button and every
      // section's own "Close" still work.
      closeOnOutside: false,
      onClose: () => {
        if (activeCleanup) { activeCleanup.destroy?.(); activeCleanup = null; }
        if (current === handle) current = null;
      },
    });
    // A modal sizes to its content by default, which is right for a dialogue
    // and wrong here: switching from About (a few centred rows) to Node packs
    // (a long list) visibly grew and shrank the whole window every click.
    // System Settings does not do that -- one fixed frame, and content that
    // does not fit scrolls inside it, which .cx-modal-body already does.
    modal.node.classList.add("cx-settings-modal");

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

const GB = (n) => (n == null ? null : `${n} GB`);

/**
 * Version, branch, and the machine ComfyUI runs on -- not the browser's,
 * which on a rental is a different box entirely and not the one worth
 * describing. Two independent fetches (git status, host facts), each
 * redrawing on its own arrival: one being slow or absent must not hold the
 * other back, and a fact this Mac genuinely has (Version) should not sit
 * behind a fact a CPU-only box never will (a GPU).
 */
function aboutSection(gitStatus, systemInfo) {
  return {
    id: "about", title: "About FunPack",
    subtitle: "Version, branch, and the machine this runs on.",
    keywords: "about version commit branch system cpu gpu memory disk python torch host hardware",
    icon: "◉", tone: "neutral",
    mount() {
      const mark = composer.text.lg({ text: "◉" });
      // xl, not lg: v4's own About name is 28px/bold (.cx-h-xl here), not
      // 22px/semibold -- the difference between a title and a wordmark.
      const name = composer.header.xl({ text: "FunPack" });
      const codename = composer.text.sm({ text: "" });
      const sub = composer.text.sm({ text: "" });
      const facts = composer.region.stack({ gap: "md" });
      const copyright = composer.hint.default({ text: "© 2025–2026 DigitalGarbage" });

      const stack = composer.region.stack({ gap: "md" });
      stack.node.classList.add("cx-about");

      let git = null;
      let sys = null;

      const row = (label, value) => composer.settingsRow.default({
        label, control: composer.text.sm({ text: value || "—" }) });
      const group = (label, rows) => [composer.label.section({ text: label }),
        composer.region.stack({ gap: "none", children: rows })];

      function render() {
        codename.node.hidden = !(git && git.codename);
        if (git && git.codename) codename.setText(`“${git.codename}”`);
        sub.setText("Cutting Room");

        const identity = composer.region.stack({ gap: "none", children: [
          row("Version", git && (git.version || "unknown")),
          row("Commit", git && git.ok ? git.commit + (git.dirty ? " (local changes)" : "") : null),
          row("Branch", git && git.ok ? git.branch : null),
        ] });

        const groups = [];
        if (sys) {
          const cpu = sys.cpu || {};
          const cores = cpu.cores && cpu.threads && cpu.cores !== cpu.threads
            ? `${cpu.cores}C/${cpu.threads}T` : (cpu.threads ? `${cpu.threads}C` : null);
          const mem = sys.memory || {};
          const gpus = sys.gpus || [];

          groups.push(...group("Hardware", [
            row("Chip", [cpu.name, cores].filter(Boolean).join(" · ") || cpu.arch),
            row("Memory", mem.available_gb != null && mem.total_gb != null
              ? `${mem.available_gb} GB free of ${mem.total_gb} GB` : GB(mem.total_gb)),
            ...(gpus.length
              ? gpus.map((g, i) => row(gpus.length > 1 ? `Graphics ${i}` : "Graphics",
                  [g.name, GB(g.vram_gb), g.capability].filter(Boolean).join(" · ")))
              : [row("Graphics", sys.mps ? "Apple GPU (MPS)" : "CPU only (no CUDA device)")]),
            row("Storage", sys.disk && sys.disk.free_gb != null && sys.disk.total_gb != null
              ? `${sys.disk.free_gb} GB available of ${sys.disk.total_gb} GB` : GB(sys.disk && sys.disk.total_gb)),
          ]));

          const t = sys.torch || {};
          groups.push(...group("Software", [
            row("System", sys.os),
            row("ComfyUI", sys.comfyui),
            row("Python", sys.python),
            row("Torch", [t.version, t.cuda ? `CUDA ${t.cuda}` : null].filter(Boolean).join(" · ")),
            ...(t.attention ? [row("Attention", t.attention)] : []),
            ...(sys.host ? [row("Host", sys.host)] : []),
          ]));
        }
        facts.set([identity, ...groups]);
      }

      render();
      stack.set([mark, name, codename, sub, facts, copyright]);

      if (gitStatus) {
        gitStatus().then((s) => { git = s; render(); }).catch(() => {
          git = { ok: false }; render();
        });
      }
      if (systemInfo) {
        systemInfo().then((s) => { sys = s; render(); }).catch(() => {});
      }
      return stack;
    },
  };
}
