// ◆ Composer: the floating "prompt craft" window. Opens from the timeline header; its tabs are the five v4 had.
import { composer as c } from "../../composer/composer.js";

const TABS = [{ value: "story", label: "Story" }, { value: "shortcuts", label: "Shortcuts" }, { value: "cuts", label: "Cuts" }, { value: "enhance", label: "Enhance" }, { value: "chat", label: "Chat" }];
const inert = (label, tone = "ghost") => c.button.sm({ label, tone, disabled: true });
const later = (what) => c.emptyState.default({ icon: "◌", title: "Not built yet", hint: what });

const story = (p) => c.region.stack({ gap: "sm", children: [
  c.toolbar.default({ items: [c.select.sm({ label: "Templates", disabled: true, options: [{ value: "", label: "Templates…" }], value: "" })], trailing: [inert("Save")] }),
  c.toolbar.default({ items: [c.label.section({ text: "Story" })], trailing: [inert("+ Add shortcut"), inert("💡")] }),
  c.textarea.md({ label: "Story", rows: 14, value: p.scenes.map((s) => s.text || "").join("\n\n") }),
  c.hint.default({ text: "Edits apply to the scenes as you type, and scene edits show here. The anchor is its own field. Shortcuts expand at generation time." }),
  c.collapsible.default({ label: "+ Variables", body: c.hint.default({ text: "$name → text, filled in at generation." }) }),
] });

const sheets = { story, shortcuts: () => later("Your trigger → replacement library."), cuts: () => later("Where a story splits into shots."),
  enhance: () => later("Rewrite a prompt with a language model."), chat: () => later("Talk a scene through with the enhancer.") };

export default {
  id: "composer",
  mount: "timeline.status",
  needs: ["project"],
  setup({ host, app }) {
    let win = null;
    const open = () => {
      if (win) return;
      const body = c.region.stack({ gap: "sm", fill: true });
      const show = (tab) => body.set([c.tabs.underline({ label: "Composer", tabs: TABS, value: tab, onChange: show }), sheets[tab](app.project)]);
      show("story");
      win = c.floating.window({ id: "composer", title: "Composer", subtitle: "prompt craft", body, width: 420, height: 520, x: 320, y: 90,
        onClose: () => { win = null; } });
    };
    host.append(c.button.sm({ label: "◆ Composer", tone: "neutral", onClick: open }).node);
  },
};
