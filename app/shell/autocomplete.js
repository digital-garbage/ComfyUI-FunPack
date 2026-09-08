// Shortcut autocomplete: as the user types in a prompt textarea, suggest
// matching shortcut triggers from the library (shortcuts.js).
//
// Positioning/dismiss/z-index is composer's own `composer.autocomplete.default`
// (composer/elements/popover.js) -- it is the ONE overlay in this kit built to
// outrank a modal (the prompt field lives inside the Constructor window), a
// z-layer rung `popover.anchored` does not claim. What stays custom here is
// the part that primitive does not offer: its `source(fieldValue)` contract
// hands a caller the WHOLE field text, and this needs to match a TRAILING
// WORD-RUN under the caret instead -- so `source` ignores the value it is
// given and reads the caret position off `ta` itself, and `onPick` replaces
// only that span rather than trusting the primitive to write the field.
//
// attach(textarea) -- safe to call once per field; a second call is a no-op.

import { composer } from "../composer/composer.js";
import { fetchAll } from "./shortcuts.js";

const DELIMS = /[,\n;]/;
const MIN_QUERY = 2;

/** Every (trigger, shortcut) pair -- one shortcut with several triggers is
 * searchable by each of them. */
function triggerIndex(items) {
  const out = [];
  for (const s of items) {
    if (s.enabled === false) continue;
    for (const t of s.triggers || []) {
      const trig = String(t || "").trim();
      if (trig) out.push({ trigger: trig, sc: s });
    }
  }
  return out;
}

/** Triggers containing `query` (substring match), prefix matches ranked
 * first, identical triggers de-duped. */
function matchTriggers(items, query) {
  const q = query.toLowerCase();
  const scored = [];
  for (const e of triggerIndex(items)) {
    const tl = e.trigger.toLowerCase();
    const at = tl.indexOf(q);
    if (at < 0) continue;
    scored.push({ ...e, rank: at === 0 ? 0 : 1, pos: at });
  }
  scored.sort((a, b) => a.rank - b.rank || a.pos - b.pos || a.trigger.length - b.trigger.length);
  const seen = new Set();
  const uniq = [];
  for (const e of scored) {
    const k = e.trigger.toLowerCase();
    if (seen.has(k)) continue;
    seen.add(k);
    uniq.push(e);
  }
  return uniq;
}

/** What to suggest under the caret: the longest trailing word-run of the
 * current token that matches any trigger, so a suggestion surfaces mid-prose
 * without needing to retype from the last comma. */
function suggestionsFor(ta, items) {
  const caret = ta.selectionStart;
  if (caret == null || caret !== ta.selectionEnd) return null;
  const text = ta.value.slice(0, caret);
  let dStart = caret;
  while (dStart > 0 && !DELIMS.test(text[dStart - 1])) dStart--;
  const starts = [];
  let prevWs = true;
  for (let i = dStart; i < caret; i++) {
    const ws = /\s/.test(text[i]);
    if (!ws && prevWs) starts.push(i);
    prevWs = ws;
  }
  for (const start of starts) {
    const run = text.slice(start, caret);
    if (run.trim().length < MIN_QUERY) continue;
    const found = matchTriggers(items, run);
    if (found.length) return { span: { start, end: caret }, items: found };
  }
  return null;
}

export function attach(ta) {
  if (!ta || ta._acAttached) return;
  ta._acAttached = true;

  // The span the CURRENT suggestion list matches, set by source() and read
  // by onPick() -- source's own return value carries the item labels, but
  // composer's autocomplete primitive has no notion of "which slice of the
  // field this item replaces", so that travels beside it instead.
  let span = null;

  // composer's autocomplete calls `source(fieldValue)` and uses its return
  // value directly (not awaited), so it has to be synchronous -- the library
  // itself, fetched over HTTP, cannot be. Kept warm here instead: fetched
  // once up front and refetched (fire-and-forget) on every focus, so typing
  // itself never waits on a network round-trip.
  let library = [];
  const warm = () => { fetchAll().then((l) => { library = l; }); };
  warm();
  ta.addEventListener("focus", warm);

  const overlay = composer.autocomplete.default({
    input: ta,
    // minChars gates on the WHOLE field's length in the primitive's own
    // contract; that is the wrong length to gate on (a long prompt with a
    // two-letter trailing word must still suggest), so it is left at 0 and
    // source() below does its own gating on the trailing word instead.
    minChars: 0,
    source: () => {
      if (!library.length) { span = null; return []; }
      const res = suggestionsFor(ta, library);
      if (!res) { span = null; return []; }
      span = res.span;
      return res.items.map((e) => ({
        label: e.trigger,
        hint: [(e.sc.replacements || [])[0], [e.sc.category, e.sc.sub_category].filter(Boolean).join(" · ")]
          .filter(Boolean).join("  ·  "),
        trigger: e.trigger,
      }));
    },
    onPick: (item) => {
      if (!span) return;
      const v = ta.value;
      const after = v.slice(span.end);
      // Finish with a trailing space so the caret is ready for the next
      // trigger and the menu does not re-open on the just-completed one --
      // except when a space/delimiter already follows, end-of-text included.
      const sep = /^(\s|[,;])/.test(after) ? "" : " ";
      ta.value = v.slice(0, span.start) + item.trigger + sep + after;
      const caret = span.start + item.trigger.length + sep.length;
      ta.setSelectionRange(caret, caret);
      ta.dispatchEvent(new Event("input", { bubbles: true }));
      ta.focus();
    },
  });

  // composer's own popover z-layer already handles outside-click and Escape
  // dismissal; nothing here needs to duplicate that.
  ta.addEventListener("blur", () => setTimeout(() => overlay.close(), 120));
}
