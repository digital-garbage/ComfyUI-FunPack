// Shortcut autocomplete: as the user types in a prompt field, suggest matching
// shortcut triggers. Toggled by the "Prompt autocomplete" editor setting.
// Reusable: ShortcutAutocomplete.attach(textarea) — safe to call once per field.
(function () {
  const { el } = window.dom;
  const S = window.Store;

  const DELIMS = /[,\n;]/;          // token boundaries — a trigger lives between delimiters
  const MIN_QUERY = 2;

  function enabled() { return !!S.getEditorSetting("autocomplete"); }

  // Every (trigger, shortcut) pair — a shortcut with multiple triggers is searchable by each.
  function triggerIndex() {
    const out = [];
    (S.get().shortcuts || []).forEach((s) => {
      if (s.enabled === false) return;
      (s.triggers || []).forEach((t) => {
        const trig = String(t || "").trim();
        if (trig) out.push({ trigger: trig, sc: s });
      });
    });
    return out;
  }

  // Triggers whose text contains `query` (substring, so "@golden" matches "golden"),
  // prefix matches ranked first, identical triggers de-duped. Unbounded — the menu itself
  // scrolls (max-height in CSS), so a prefix shared by many shortcuts (e.g. 15 "lazy_*"
  // triggers) shows every one of them instead of silently hiding the tail.
  function matchTriggers(query) {
    const q = query.toLowerCase();
    const scored = [];
    for (const e of triggerIndex()) {
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
      seen.add(k); uniq.push(e);
    }
    return uniq;
  }

  // What to suggest under the caret. The query is a TRAILING WORD-RUN of the current
  // token (text since the last delimiter): we try the longest run first ("golden ho"),
  // then shorter tails ("ho"), and use the longest run that matches any trigger. This
  // means suggestions surface mid-prose (you don't have to retype from a comma), while
  // the replaced span stays a clean unit so accepting can't duplicate words.
  function suggestionsFor(ta) {
    const caret = ta.selectionStart;
    if (caret == null || caret !== ta.selectionEnd) return null;
    const text = ta.value.slice(0, caret);
    let dStart = caret;
    while (dStart > 0 && !DELIMS.test(text[dStart - 1])) dStart--;
    // Word-run start offsets within the token, in order (longest run → shortest).
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
      const found = matchTriggers(run);
      if (found.length) return { span: { start, end: caret }, items: found };
    }
    return null;
  }

  function attach(ta) {
    if (!ta || ta._acAttached) return;
    ta._acAttached = true;

    let menu = null;
    let items = [];
    let active = -1;
    let span = null;

    function close() {
      if (menu) { menu.remove(); menu = null; }
      items = []; active = -1; span = null;
    }

    function accept(i) {
      if (!span || !items[i]) return;
      const trig = items[i].trigger;
      const v = ta.value;
      const after = v.slice(span.end);
      // Finish the shortcut with a trailing space so the caret is ready for the next one
      // AND the menu doesn't re-open on the just-completed trigger (it would match itself).
      // This includes end-of-text — skip the space ONLY when a space/delimiter already
      // follows. (An earlier `$` here skipped the space at end-of-text, the common case,
      // which made the menu stick on the accepted trigger.)
      const sep = /^(\s|[,;])/.test(after) ? "" : " ";
      ta.value = v.slice(0, span.start) + trig + sep + after;
      const caret = span.start + trig.length + sep.length;
      ta.setSelectionRange(caret, caret);
      close();
      ta.dispatchEvent(new Event("input", { bubbles: true }));
      ta.focus();
    }

    function paintActive() {
      if (!menu) return;
      [...menu.children].forEach((c, i) => c.classList.toggle("active", i === active));
      const cur = menu.children[active];
      if (cur) cur.scrollIntoView({ block: "nearest" });
    }

    function open(found, sp) {
      close();
      // One menu at a time across the whole app — sweep away any orphan left behind by a
      // re-rendered/removed textarea (otherwise it strands in the corner with no dismiss).
      document.querySelectorAll(".ac-menu").forEach((m) => m.remove());
      if (!ta.isConnected) return;
      span = sp; items = found; active = 0;
      menu = el("div", "ac-menu");
      found.forEach((e, i) => {
        const row = el("div", "ac-item" + (i === 0 ? " active" : ""));
        row.append(el("span", "ac-trigger", e.trigger));
        const rep = (e.sc.replacements || [])[0];
        if (rep) row.append(el("span", "ac-prompt", rep));
        const tag = [e.sc.category, e.sc.sub_category].filter(Boolean).join(" · ");
        if (tag) row.append(el("span", "ac-cat", tag));
        row.addEventListener("mousedown", (ev) => { ev.preventDefault(); accept(i); });
        row.addEventListener("mouseenter", () => { active = i; paintActive(); });
        menu.append(row);
      });
      document.body.append(menu);
      position();
    }

    function position() {
      if (!menu) return;
      if (!ta.isConnected) { close(); return; }
      const r = ta.getBoundingClientRect();
      // Flip above the field when there isn't room below (e.g. Easy Gen's prompt box
      // sits in a footer bar at the bottom of the viewport) — otherwise the menu opens
      // off-screen under the field with no way to see or click any suggestion.
      const GAP = 2;
      const menuH = menu.offsetHeight || 240;
      const spaceBelow = window.innerHeight - r.bottom;
      const spaceAbove = r.top;
      const openAbove = spaceBelow < menuH + GAP && spaceAbove > spaceBelow;
      menu.style.left = r.left + "px";
      if (openAbove) {
        menu.style.top = "";
        menu.style.bottom = (window.innerHeight - r.top + GAP) + "px";
        menu.style.maxHeight = Math.max(120, Math.min(240, spaceAbove - GAP * 2)) + "px";
      } else {
        menu.style.bottom = "";
        menu.style.top = (r.bottom + GAP) + "px";
        menu.style.maxHeight = Math.max(120, Math.min(240, spaceBelow - GAP * 2)) + "px";
      }
      menu.style.minWidth = Math.min(r.width, 420) + "px";
      menu.style.maxWidth = Math.max(r.width, 320) + "px";
    }

    function refresh() {
      if (!enabled() || !ta.isConnected) { close(); return; }
      const res = suggestionsFor(ta);
      if (!res) { close(); return; }
      open(res.items, res.span);
    }

    ta.addEventListener("input", refresh);
    ta.addEventListener("click", refresh);
    ta.addEventListener("keydown", (e) => {
      if (!menu) return;
      if (e.key === "ArrowDown") { e.preventDefault(); active = (active + 1) % items.length; paintActive(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); active = (active - 1 + items.length) % items.length; paintActive(); }
      else if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); accept(active); }
      else if (e.key === "Escape") { e.preventDefault(); close(); }
    });
    ta.addEventListener("blur", () => setTimeout(close, 120));
    window.addEventListener("scroll", () => { if (menu) position(); }, true);
    // A panel re-render can remove this textarea while a menu is open; removal doesn't
    // fire blur reliably, so watch the DOM and close (+ stop watching) when it detaches.
    const detachObs = new MutationObserver(() => {
      if (!ta.isConnected) { close(); detachObs.disconnect(); }
    });
    detachObs.observe(document.body, { childList: true, subtree: true });
  }

  window.ShortcutAutocomplete = { attach };
})();
