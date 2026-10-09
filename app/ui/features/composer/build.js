// Build: a scene drafted from your own shortcuts, those in scenes you liked first. Rating renders is what teaches it.
import { composer as c } from "../../composer/composer.js";
import { compose } from "../../shell/habits.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";

export function build(app, own) {
  const p = app.project;
  let alive = true, typed = false, shown, target = null, asked = 0;
  // The scene whose prompt a clip runs on: a cut's is its unit's first clip.
  const root = () => p.selected && unitRoot(p.project, genUnitId(p.selected)) || p.selected;
  const number = (sc) => p.scenes.indexOf(sc) + 1;
  const idea = c.textarea.md({ label: "Idea", rows: 3, placeholder: "Optional: a few words or triggers to start from.", onInput: () => { typed = true; } });
  const draft = c.textarea.md({ label: "Draft", rows: 4, value: "", placeholder: "Press Build.", onCommit: () => preview() });
  const said = c.hint.default({ text: "" }), full = c.text.sm({ text: "" });
  // The idea follows the selected scene until you type in it; another scene starts it again.
  const follow = () => { const sc = root(); if ((sc && sc.id) !== shown) typed = false; shown = sc && sc.id; if (!typed) idea.setValue((sc && sc.text) || ""); };      // by id: Undo puts a copy of the same scene in
  follow();
  own(app.on(() => { if (alive) follow(); }));
  own(() => { alive = false; });

  async function preview() {       // what Generate would send for the draft, with one draw of each shortcut
    const mine = ++asked, sc = target && p.scenes.find((s) => s.id === target);
    const prefix = (app.promptPrefix || []).flatMap((f) => { try { return f(sc); } catch { return []; } });
    try {
      const r = await app.api.expandPrompt({ text: draft.value, anchor: [...prefix, p.anchor].filter((x) => x && String(x).trim()).join(" "), postfix: p.postfix, postfix_enabled: p.postfixEnabled, variables: p.variables });
      if (alive && mine === asked) full.setText(r.text || "");
    } catch (err) { if (alive && mine === asked) full.setText(`Could not build the full prompt: ${err.message}`); }
  }
  const go = async () => {
    const sc = root();
    if (!sc || !isGenerative(sc)) return said.setText(sc ? "This clip is an imported video: it has no prompt." : "Select a scene first.");
    const words = idea.value;                   // as it reads at the click: the selection may change while this works
    button.setBusy(true);
    try {
      await p.flush();                          // a rating or edit made a moment ago is in the projects the ratings are read from
      const [library, stats] = await Promise.all([app.api.shortcuts().then((r) => r.shortcuts || []), app.api.suggestionStats()]);
      if (!alive) return;
      const { text, added } = compose(library, stats, words);
      target = sc.id;
      draft.setValue(text);
      said.setText(`For scene ${number(sc)}. ${added.length ? `Added: ${added.map((s) => s.name).join(", ")}` : library.length ? "Nothing to add: every category is already used." : "No shortcuts yet: add some in the Shortcuts tab."}`);
      preview();
    } catch (err) { if (alive) said.setText(err.message); }
    finally { if (alive) button.setBusy(false); }
  };
  const use = () => {
    const sc = target && p.scenes.find((s) => s.id === target);
    if (!sc || !draft.value.trim()) return said.setText(sc ? "The draft is empty." : "Press Build first.");
    p.setText(sc.id, draft.value);
    said.setText(`Put in scene ${number(sc)}. Edit > Undo takes it back.`);
  };
  const button = c.button.sm({ label: "Build", tone: "primary", onClick: go });
  return c.region.stack({ gap: "sm", children: [
    c.hint.default({ text: "Drafts a scene from your shortcuts. Each like makes one come up more often; each dislike less, and dislikes in a row much less." }),
    idea,
    c.toolbar.default({ items: [button] }),
    draft, said,
    c.field.default({ label: "Full prompt", hint: "One draw: a shortcut with several choices picks again each run.", control: full }),
    c.toolbar.default({ items: [c.button.sm({ label: "Use in scene", tone: "neutral", onClick: use })] }),
  ] });
}
