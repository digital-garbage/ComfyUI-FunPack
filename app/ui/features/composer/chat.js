// Chat: comment on the enhancer's last rewrite. The next Generate rewrites again with every comment still applying; Reset starts over.
// The comments are the project's own (editor_settings.enhance_chat) and reach the enhancer node with the run.
import { composer as c } from "../../composer/composer.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";

export function chat(app, own) {
  const p = app.project, page = c.region.stack({ gap: "sm" });
  let last = null, alive = true;
  own(() => { alive = false; });
  const rounds = () => { const r = p.pref("enhance_chat", []); return Array.isArray(r) ? r : []; };
  const typed = () => {                                   // the prompt a comment is about: the picked scene's own text, as typed
    const open = p.project, sc = open && (open.scenes.find((s) => s.id === app.selection.focus) || open.scenes.find(isGenerative));
    const root = sc && isGenerative(sc) ? unitRoot(open, genUnitId(sc)) : null;
    return root ? (root.text || "").trim() : undefined;
  };
  const first = (r) => Object.values(r.rewrites || {})[0];
  let draft = "";
  function draw() {
    if (!alive) return;
    if (!p.project) return page.set([c.hint.default({ text: "Open a project first." })]);
    const list = rounds(), ok = last && last.ok;
    const bubbles = [...list.flatMap((r) => [first(r) ? c.text.sm({ text: `Model: ${first(r)}` }) : null, c.text.sm({ text: `You: ${r.comment}` })]),
      ...(ok ? [c.hint.default({ text: `Last rewrite of: ${String(last.before || "").slice(0, 80)}` }), c.text.sm({ text: `Model: ${last.after}` })] : []),
      ...(!list.length && !ok ? [c.hint.default({ text: "Generate once with the enhancer on, then comment here." })] : [])].filter(Boolean);
    const box = c.textarea.md({ label: "Comment", rows: 2, value: draft, placeholder: "e.g. warmer light, no rain", onInput: (v) => { draft = v; } });
    const send = () => {
      const comment = draft.trim();
      if (!comment) return;
      const prev = list.length ? first(list[list.length - 1]) : null;
      const fresh = ok && last.after !== prev;                 // a rewrite nobody has commented on yet
      const original = typed();
      const round = { rewrites: fresh ? { whole: last.after } : {}, comment, ...(original === undefined ? {} : { original }) };
      draft = "";
      p.setPref("enhance_chat", [...list, round]);
      draw();
    };
    page.set([c.label.section({ text: "Chat" }), c.hint.default({ text: "Comment on the last rewrite. The next Generate rewrites again with every comment still applying. Reset starts over." }),
      c.region.stack({ gap: "xs", children: bubbles }), box,
      c.toolbar.default({ items: [c.button.sm({ label: "Send", tone: "primary", onClick: send })], trailing: [c.button.sm({ label: "Reset", tone: "ghost", disabled: !list.length, onClick: () => { p.setPref("enhance_chat", []); draw(); } })] })]);
  }
  draw();
  app.api.enhancerRuns().then((r) => { last = (r.runs || []).slice(-1)[0] || null; draw(); }).catch(() => {});
  return page;
}
