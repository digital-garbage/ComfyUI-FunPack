// The Settings window. It owns no section: features announce theirs by mounting at "settings"
//   setup({ host }) { host.add({ id, title, subtitle, keywords, icon, tone, mount({ setFooter, close }) -> handle }) }
// and the window lists whatever arrived. It is also the one place a "Settings" button is made.
import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";

export function createSettings({ menubar }) {
  const sections = [];
  // childNodes: the feature loader checks a region's children to undo a feature that fails half-way
  const host = { childNodes: [], add: (section) => sections.push(section) };
  offer("settings", host);
  let current = null;

  function open(id) {
    if (current) { current.show(id); return current; }
    let shown = null;
    const content = composer.region.stack({ gap: "sm", fill: true });
    content.node.classList.add("cx-settings-content");

    function show(pick) {
      const spec = sections.find((s) => s.id === pick) || sections[0];
      if (!spec) return;
      shown?.destroy?.();
      modal.setFooter({});                  // a section with no footer must not keep the last one's buttons
      shown = spec.mount({ setFooter: (f) => modal.setFooter(f), close: () => modal.close("done") });
      content.set([composer.header.md({ text: spec.title }),
        spec.subtitle ? composer.hint.default({ text: spec.subtitle }) : null, shown]);
      nav.setValue(spec.id);
    }

    const nav = composer.filterList.md({
      items: sections.map((s) => ({ id: s.id, label: s.title, icon: s.icon, keywords: s.keywords, tone: s.tone || "neutral" })),
      value: id, onChange: show, placeholder: "Search settings",
    });
    nav.node.classList.add("cx-filter-full", "cx-settings-rail");
    const body = composer.region.stack({ gap: "none", fill: true, children: [nav, content] });
    body.node.classList.add("cx-settings-body");
    const modal = composer.modal.generic({
      title: "Settings", size: "xl", body,
      closeOnOutside: false,                // a half-finished edit must not vanish on a stray backdrop click
      onClose: () => { shown?.destroy?.(); current = null; },
    });
    modal.node.classList.add("cx-settings-modal");
    current = { show, close: () => modal.close() };
    show(id);
    return current;
  }

  // Added once the other menus are in, so it sits last as in v4.
  const addButton = () => menubar.append(composer.button.sm({ label: "Settings", onClick: () => open() }).node);
  return { open, addButton, sections: () => [...sections] };
}
