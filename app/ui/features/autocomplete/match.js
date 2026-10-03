// Which shortcut triggers fit the word under the caret. Pure.
const DELIMS = /[,\n;()[\]]/;       // a trigger lives between delimiters; brackets wrap one: (trigger:1.5), [trigger@1-3]
const MIN_QUERY = 2;

/** Triggers containing `query` (prefix matches first), one row per distinct trigger. `library`: the shortcuts. */
export function matchTriggers(library, query) {
  const q = query.toLowerCase(), seen = new Set(), scored = [];
  for (const sc of library) {
    if (sc.enabled === false) continue;
    for (const raw of sc.triggers || []) {
      const trigger = String(raw || "").trim(), at = trigger.toLowerCase().indexOf(q);
      if (trigger && at >= 0) scored.push({ trigger, sc, rank: at === 0 ? 0 : 1, at });
    }
  }
  scored.sort((a, b) => a.rank - b.rank || a.at - b.at || a.trigger.length - b.trigger.length);
  return scored.filter((e) => !seen.has(e.trigger.toLowerCase()) && seen.add(e.trigger.toLowerCase()));
}

/** What to offer under the caret: the longest trailing run of words in the current token that matches any trigger.
 *  -> { span: {start, end}, items } or null. */
export function suggestionsAt(library, text, caret) {
  const head = text.slice(0, caret);
  let from = caret;
  while (from > 0 && !DELIMS.test(head[from - 1])) from -= 1;
  const starts = [];
  for (let i = from, space = true; i < caret; i += 1) { const ws = /\s/.test(head[i]); if (!ws && space) starts.push(i); space = ws; }
  for (const start of starts) {
    const run = head.slice(start, caret);
    if (run.trim().length < MIN_QUERY) continue;
    const items = matchTriggers(library, run);
    if (items.length) return { span: { start, end: caret }, items };
  }
  return null;
}

/** `text` with the span replaced by the trigger and a space after it (none when a separator already follows). -> { text, caret } */
export function accept(text, span, trigger) {
  const after = text.slice(span.end), sep = /^(\s|[,;)\]:@])/.test(after) ? "" : " ";
  return { text: text.slice(0, span.start) + trigger + sep + after, caret: span.start + trigger.length + sep.length };
}
