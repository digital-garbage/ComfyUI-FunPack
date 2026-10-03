// ◆ Composer: the floating "prompt craft" window. Opens from the timeline header; its tabs are the five v4 had.
import { composer as c } from "../../composer/composer.js";
import { applyStory, clash, joinStory } from "./story.js";
import { shortcuts } from "./shortcuts.js";
import { cuts } from "./cuts.js";
import { enhance } from "./enhance.js";
import { pickShortcut, templatesBar, variablesPanel } from "./storytools.js";

const TABS = [{ value: "story", label: "Story" }, { value: "shortcuts", label: "Shortcuts" }, { value: "cuts", label: "Cuts" }, { value: "enhance", label: "Enhance" }, { value: "chat", label: "Chat" }];
const inert = (label, tone = "ghost") => c.button.sm({ label, tone, disabled: true });
const later = (what) => c.emptyState.default({ icon: "◌", title: "Not built yet", hint: what });

// The box follows the scenes; typing in it rewrites them (after a short pause, or when focus leaves).
const story = (app, own) => {
  const p = app.project;
  let marker = "qcut", words = ["qcut"], timer = 0, asked = 0;
  const box = c.textarea.md({ label: "Story", rows: 14, value: p.project ? joinStory(p.project, marker) : "", onInput: (v) => { clearTimeout(timer); timer = setTimeout(() => { timer = 0; apply(v); }, 700); }, onCommit: (v) => { clearTimeout(timer); timer = 0; apply(v); } });
  const area = box.node, warn = c.hint.default({ text: "" });
  async function apply(text) {
    if (!p.project) return;
    const mine = ++asked, pid = p.project.id;
    try {
      const { scenes } = await app.api.storySplit(text);
      if (mine !== asked || !p.project || p.project.id !== pid) return;     // typed more meanwhile, or another project opened: not for this one
      p.edit((pr) => applyStory(pr, scenes));
    } catch (err) { c.toast.warn({ text: `Could not split the story: ${err.message}` }); }
  }
  app.api.storyMarkers().then((r) => { words = r.markers || words; marker = words[0]; sync(); }).catch(() => {});
  function sync() {
    warn.setText(p.project && clash(p.project, words) ? `A scene's own text contains a cut word (${words.join(", ")}), so the story shows it as two scenes. Remove the word from that scene.` : "");
    if (!p.project || document.activeElement === area || timer) return;           // never rewrite the box while it is being typed in
    const next = joinStory(p.project, marker);
    if (area.value !== next) box.setValue(next);
  }
  own(app.on(sync));
  sync();
  return c.region.stack({ gap: "sm", children: [
    templatesBar(app, own),
    c.toolbar.default({ items: [c.label.section({ text: "Story" })], trailing: [c.button.sm({ label: "+ Add shortcut", tone: "ghost", title: "Insert a shortcut trigger: browse the library", onClick: async () => {
      const trigger = await pickShortcut(app);
      if (!trigger) return;
      area.focus(); area.setRangeText(`${area.selectionStart && !/\s$/.test(area.value.slice(0, area.selectionStart)) ? " " : ""}${trigger} `, area.selectionStart, area.selectionEnd, "end");
      area.dispatchEvent(new Event("input", { bubbles: true }));
    } }), inert("💡")] }),
    box, warn,
    c.hint.default({ text: `Scenes are cut at the word “${marker}”. Edits apply to the scenes as you type; the anchor is its own field.` }),
    c.collapsible.default({ label: "+ Variables", body: variablesPanel(app) }),
  ] });
};

const sheets = { story, shortcuts, cuts,
  enhance, chat: () => later("Talk a scene through with the enhancer.") };

export default {
  id: "composer",
  mount: "timeline.status",
  needs: ["project", "api", "pipeline"],
  setup({ host, app }) {
    let win = null, owned = [];
    const cleanup = () => { owned.forEach((f) => f()); owned = []; };
    const open = () => {
      if (win) return;
      const body = c.region.stack({ gap: "sm", fill: true });
      const show = (tab) => { cleanup(); body.set([c.tabs.underline({ label: "Composer", tabs: TABS, value: tab, onChange: show }), sheets[tab](app, (off) => owned.push(off))]); };
      show("story");
      win = c.floating.window({ id: "composer", title: "Composer", subtitle: "prompt craft", body, width: 420, height: 520, x: 320, y: 90,
        onClose: () => { cleanup(); win = null; } });
    };
    host.append(c.button.sm({ label: "◆ Composer", tone: "neutral", onClick: open }).node);
    return cleanup;
  },
};
