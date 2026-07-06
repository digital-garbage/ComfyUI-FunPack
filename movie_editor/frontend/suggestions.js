// Shortcut suggestions ("Ideas" bulb): a quiet nudge next to prompt fields that lights
// up when whole shortcut categories (camera moves, sub-actions, …) exist in the library
// but are absent from the current prompt — easy to forget, and the video comes out plain.
// Never opens on its own: click the bulb for a handful of insertable ideas, ⟳ rerolls.
// Toggled by the "Shortcut ideas" editor setting. Reusable: ShortcutSuggest.bulb(textarea).
(function () {
  const { el } = window.dom;
  const S = window.Store;

  const MAX_CHIPS = 3;            // random samples shown per category (before "+N more")

  function enabled() { return !!S.getEditorSetting("suggestions"); }

  function libraryShortcuts() {
    return (S.get().shortcuts || []).filter((s) => s.enabled !== false && (s.triggers || []).length);
  }

  // Does `text` contain `trigger` as a whole token (word boundaries on both sides)?
  // Triggers may contain spaces, so this is a boundary-checked substring scan.
  function containsTrigger(textLc, trigger) {
    const t = trigger.toLowerCase();
    if (!t) return false;
    let at = textLc.indexOf(t);
    const isWord = (ch) => /[a-z0-9_]/.test(ch || "");
    while (at >= 0) {
      const before = textLc[at - 1];
      const after = textLc[at + t.length];
      if (!isWord(before) && !isWord(after)) return true;
      at = textLc.indexOf(t, at + 1);
    }
    return false;
  }

  // Group the library by category and split into covered (some trigger appears in the
  // prompt) vs missing. The missing groups are the "you forgot about these" pool.
  function analyze(text) {
    const textLc = String(text || "").toLowerCase();
    const groups = new Map();      // category -> { used: bool, items: [shortcut] }
    for (const sc of libraryShortcuts()) {
      const cat = sc.category || "other";
      if (!groups.has(cat)) groups.set(cat, { used: false, items: [] });
      const g = groups.get(cat);
      g.items.push(sc);
      if (!g.used && (sc.triggers || []).some((t) => containsTrigger(textLc, String(t || "").trim()))) {
        g.used = true;
      }
    }
    const missing = [...groups.entries()].filter(([, g]) => !g.used)
      .map(([cat, g]) => ({ cat, items: g.items }));
    return { missing, total: groups.size };
  }

  function shuffled(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  // Insert at the caret (replacing any selection), padded with spaces against
  // neighboring words, and fire `input` so the field's normal handlers persist it.
  function insertIntoTextarea(ta, text) {
    const v = ta.value;
    let s = ta.selectionStart, e = ta.selectionEnd;
    if (s == null) { s = e = v.length; }
    const before = v.slice(0, s), after = v.slice(e);
    const lead = before && !/\s$/.test(before) ? " " : "";
    const trail = !/^(\s|[,;])/.test(after) ? " " : "";
    const ins = lead + text + trail;
    ta.value = before + ins + after;
    const caret = s + ins.length;
    ta.setSelectionRange(caret, caret);
    ta.dispatchEvent(new Event("input", { bubbles: true }));
    ta.focus();
  }

  function bulb(ta) {
    const btn = el("button", "btn ghost tiny sugg-bulb", "💡");
    btn.type = "button";
    if (!enabled()) { btn.style.display = "none"; return btn; }

    let pop = null;
    let unlisten = null;           // detaches the document-level dismiss listeners
    function closePop() {
      if (pop) { pop.remove(); pop = null; }
      if (unlisten) { unlisten(); unlisten = null; }
    }

    function refresh() {
      const { missing } = analyze(ta.value);
      const n = missing.length;
      btn.classList.toggle("on", n > 0 && !!ta.value.trim());
      btn.title = n > 0
        ? `Ideas — ${n} shortcut ${n === 1 ? "category" : "categories"} this prompt isn't using`
        : "Ideas — this prompt already touches every shortcut category";
    }

    function position() {
      if (!pop) return;
      const r = btn.getBoundingClientRect();
      pop.style.top = (r.bottom + 4) + "px";
      pop.style.left = Math.max(8, Math.min(r.left, window.innerWidth - 340)) + "px";
    }

    function chipRow(sc) {
      const trig = String((sc.triggers || [])[0] || "").trim();
      if (!trig) return null;
      const row = el("div", "sugg-item");
      row.append(el("span", "ac-trigger", trig));
      const rep = (sc.replacements || [])[0];
      if (rep) row.append(el("span", "ac-prompt", rep));
      if (sc.sub_category) row.append(el("span", "ac-cat", sc.sub_category));
      row.title = rep || trig;
      // mousedown (not click) so the textarea keeps focus context, same as autocomplete.
      row.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        insertIntoTextarea(ta, trig);
        closePop();
        refresh();
      });
      return row;
    }

    function openPop() {
      closePop();
      document.querySelectorAll(".sugg-pop").forEach((p) => p.remove());
      // Never open off an invisible/detached bulb — the popover would strand at (0,0).
      const br = btn.getBoundingClientRect();
      if (!btn.isConnected || (!br.width && !br.height)) return;

      const { missing } = analyze(ta.value);
      pop = el("div", "sugg-pop");

      const head = el("div", "sugg-head");
      head.append(el("span", "sugg-title", missing.length
        ? "Categories not in this prompt yet"
        : "All categories covered ✓"));
      const reroll = el("button", "btn ghost tiny", "⟳");
      reroll.title = "Shuffle the samples";
      reroll.onclick = (ev) => { ev.stopPropagation(); fill(); };
      head.append(reroll);
      pop.append(head);

      const list = el("div", "sugg-list");
      pop.append(list);

      // Category-grouped browser: every missing category is listed (so you SEE what
      // whole families you're skipping), with a few random samples under each and a
      // "+N more" toggle to browse that category in full.
      const expanded = new Set();
      function fill() {
        list.replaceChildren();
        if (!missing.length) {
          list.append(el("div", "sugg-empty",
            "Nothing left to suggest — every shortcut category already appears here."));
          return;
        }
        const cats = missing.slice().sort((a, b) => a.cat.localeCompare(b.cat));
        for (const m of cats) {
          const headRow = el("div", "sugg-cat-head");
          headRow.append(el("span", "sugg-cat-name", m.cat));
          headRow.append(el("span", "sugg-cat-count", String(m.items.length)));
          const showAll = expanded.has(m.cat);
          if (m.items.length > MAX_CHIPS) {
            const more = el("button", "btn ghost tiny sugg-more",
              showAll ? "less" : `+${m.items.length - MAX_CHIPS} more`);
            more.onclick = (ev) => {
              ev.stopPropagation();
              showAll ? expanded.delete(m.cat) : expanded.add(m.cat);
              fill();
            };
            headRow.append(more);
          }
          list.append(headRow);
          const items = showAll ? m.items : shuffled(m.items).slice(0, MAX_CHIPS);
          for (const sc of items) {
            const row = chipRow(sc);
            if (row) list.append(row);
          }
        }
      }
      fill();

      document.body.append(pop);
      position();
      // Dismiss on any outside press or Escape; reposition while scrolling.
      // Every close path funnels through closePop, which also detaches these.
      const onDown = (ev) => { if (pop && !pop.contains(ev.target) && ev.target !== btn) closePop(); };
      const onKey = (ev) => { if (ev.key === "Escape") closePop(); };
      const onScroll = () => position();
      document.addEventListener("mousedown", onDown, true);
      document.addEventListener("keydown", onKey, true);
      window.addEventListener("scroll", onScroll, true);
      unlisten = () => {
        document.removeEventListener("mousedown", onDown, true);
        document.removeEventListener("keydown", onKey, true);
        window.removeEventListener("scroll", onScroll, true);
      };
    }

    btn.onclick = (ev) => { ev.preventDefault(); if (pop) closePop(); else openPop(); };

    // Keep the glow honest while typing — cheap scan, debounced.
    let timer = null;
    ta.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(refresh, 400);
    });
    // Callers create the bulb before appending the field to the DOM — defer the first scan.
    setTimeout(refresh, 0);
    // A panel re-render can remove the bulb while its popover is open; close it (and
    // stop watching) when the bulb detaches, same pattern as the autocomplete menu.
    const detachObs = new MutationObserver(() => {
      if (!btn.isConnected) { closePop(); detachObs.disconnect(); }
    });
    setTimeout(() => { if (btn.isConnected) detachObs.observe(document.body, { childList: true, subtree: true }); }, 0);
    return btn;
  }

  window.ShortcutSuggest = { bulb };
})();
