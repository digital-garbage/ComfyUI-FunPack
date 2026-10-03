// Chat: comment on the enhancer's last rewrite. The next Generate rewrites again with every comment still applying; Reset starts over.
// The comments are the project's own (editor_settings.enhance_chat); a generate hook hands them to the enhancer node for the prompt they were about.
import { composer as c } from "../../composer/composer.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";

const NODE = "FunPackEnhancePrompt";

/** Generate hook: the comments about THIS typed prompt go to the enhancer slot; the others are said to be left out. */
export function chatHook({ project, scene, slots }) {
  const chat = (project.editor_settings || {}).enhance_chat, slot = (slots || []).find((s) => s.node === NODE);
  if (!slot || !Array.isArray(chat)) return null;
  const typed = (scene.text || "").trim(), mine = chat.filter((r) => r && (r.original === undefined || r.original === typed)), left = chat.length - mine.length;
  return { inputs: mine.length ? { [slot.id]: { chat: JSON.stringify(mine) } } : {}, notes: left ? [`${left} Chat comment(s) are about a different prompt and were left out of this run.`] : [] };
}

export function chat(app, own) {
  const p = app.project, page = c.region.stack({ gap: "sm" });
  let last = null, alive = true, key = "", draft = "", runs = app.lastRun.n;
  own(() => { alive = false; });
  const rounds = () => { const r = p.pref("enhance_chat", []); return Array.isArray(r) ? r : []; };
  const typed = () => {                                   // the prompt a comment is about: the picked scene's own text, as typed
    const open = p.project, sc = open && (open.scenes.find((s) => s.id === app.selection.focus) || open.scenes.find(isGenerative));
    const root = sc && isGenerative(sc) ? unitRoot(open, genUnitId(sc)) : null;
    return root ? (root.text || "").trim() : undefined;
  };
  const first = (r) => Object.values(r.rewrites || {})[0];
  function draw(force) {
    if (!alive) return;
    const list = rounds(), ok = last && last.ok, next = JSON.stringify([p.project && p.project.id, list, last && last.after]);
    if (!force && (next === key || (page.node.contains(document.activeElement) && document.activeElement.tagName === "TEXTAREA"))) return;       // not under the caret
    key = next;
    if (!p.project) return page.set([c.hint.default({ text: "Open a project first." })]);
    const bubbles = [...list.flatMap((r) => [first(r) ? c.text.sm({ text: `Model: ${first(r)}` }) : null, c.text.sm({ text: `You: ${r.comment}` })]),
      ...(ok ? [c.hint.default({ text: `Last rewrite of: ${String(last.before || "").slice(0, 80)}` }), c.text.sm({ text: `Model: ${last.after}` })] : []),
      ...(!list.length && !ok ? [c.hint.default({ text: "Generate once with the enhancer on, then comment here." })] : [])].filter(Boolean);
    const box = c.textarea.md({ label: "Comment", rows: 2, value: draft, placeholder: "e.g. warmer light, no rain", onInput: (v) => { draft = v; } });
    const send = () => {
      const comment = draft.trim(), now = rounds();
      if (!comment) return;
      const prev = now.length ? first(now[now.length - 1]) : null, here = typed();
      const fresh = ok && last.after !== prev && app.lastRun.typed !== undefined && app.lastRun.typed === here;      // a rewrite of THIS prompt that nobody has commented on yet
      draft = "";
      p.setPref("enhance_chat", [...now, { rewrites: fresh ? { whole: last.after } : {}, comment, ...(here === undefined ? {} : { original: here }) }]);
      draw(true);
    };
    page.set([c.label.section({ text: "Chat" }), c.hint.default({ text: "Comment on the last rewrite. The next Generate rewrites again with every comment still applying. Reset starts over." }),
      c.region.stack({ gap: "xs", children: bubbles }), box,
      c.toolbar.default({ items: [c.button.sm({ label: "Send", tone: "primary", onClick: send })], trailing: [c.button.sm({ label: "Reset", tone: "ghost", disabled: !list.length, onClick: () => { p.setPref("enhance_chat", []); draw(true); } })] })]);
  }
  const look = () => app.api.enhancerRuns().then((r) => { last = (r.runs || []).slice(-1)[0] || null; draw(); }).catch(() => {});
  own(app.on((what) => { if (what === "open" || app.lastRun.n !== runs) { runs = app.lastRun.n; look(); } else draw(); }));
  draw(true);
  look();
  return page;
}
