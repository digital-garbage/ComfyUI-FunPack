// Build: a scene drafted from your own shortcuts, those in scenes you liked first. Rating renders is what teaches it.
import { composer as c } from "../../composer/composer.js";
import { compose } from "../../shell/habits.js";

export function build(app, own) {
  const p = app.project;
  let alive = true;
  own(() => { alive = false; });
  const idea = c.textarea.md({ label: "Idea", rows: 3, value: (p.selected && p.selected.text) || "", placeholder: "Optional: a few words or triggers to start from." });
  const draft = c.textarea.md({ label: "Draft", rows: 4, value: "", placeholder: "Press Build.", onCommit: () => preview() });
  const said = c.hint.default({ text: "" }), full = c.text.sm({ text: "" });

  async function preview() {       // what Generate would send for the draft, with one draw of each shortcut
    const prefix = (app.promptPrefix || []).flatMap((f) => { try { return f(p.selected); } catch { return []; } });
    try {
      const r = await app.api.expandPrompt({ text: draft.value, anchor: [...prefix, p.anchor].filter((x) => x && String(x).trim()).join(" "), postfix: p.postfix, postfix_enabled: p.postfixEnabled, variables: p.variables });
      if (alive) full.setText(r.text || "");
    } catch (err) { if (alive) full.setText(`Could not build the full prompt: ${err.message}`); }
  }
  async function go() {
    let library, stats;
    try { [library, stats] = await Promise.all([app.api.shortcuts().then((r) => r.shortcuts || []), app.api.suggestionStats()]); }
    catch (err) { return said.setText(err.message); }
    if (!alive) return;
    const { text, added } = compose(library, stats, idea.value);
    draft.setValue(text);
    said.setText(added.length ? `Added: ${added.map((s) => s.name).join(", ")}` : library.length ? "Nothing to add: every category is used, or what is left was rated down." : "No shortcuts yet: add some in the Shortcuts tab.");
    preview();
  }
  const use = () => {
    if (!p.selected) return said.setText("Select a scene first.");
    p.setText(p.selectedId, draft.value);
    said.setText("Put in the selected scene. Edit > Undo takes it back.");
  };
  return c.region.stack({ gap: "sm", children: [
    c.hint.default({ text: "Drafts a scene from your shortcuts, favouring those in scenes you liked. Rate renders to teach it." }),
    idea,
    c.toolbar.default({ items: [c.button.sm({ label: "Build", tone: "primary", onClick: go })] }),
    draft, said,
    c.field.default({ label: "Full prompt", control: full }),
    c.toolbar.default({ items: [c.button.sm({ label: "Use in selected scene", tone: "neutral", onClick: use })] }),
  ] });
}
