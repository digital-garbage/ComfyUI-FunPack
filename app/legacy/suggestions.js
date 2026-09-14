// Shortcut suggestions ("Ideas" bulb): a prompt-craft co-pilot next to prompt fields.
// The bulb glows when whole shortcut categories go unused in the current prompt; the
// popover mixes what it knows about YOUR habits (mined from all saved projects — pairs
// used in the same shot, what shot usually follows the previous one) with browsing:
// continue inside categories you're already using, or jump into ones you're not.
// Never opens on its own. Toggled by the "Shortcut ideas" editor setting.
// Reusable: ShortcutSuggest.bulb(textarea, { prevText: () => "…" }) — prevText supplies
// the previous shot's prompt for "usual next shot" suggestions (scene inspector only).
(function () {
  const { el } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;

  const MAX_CHIPS = 3;            // random samples shown per category (before "+N more")
  const MAX_HABIT = 4;            // ranked rows per habit section (pairs / next shot)
  const MIN_HABIT_N = 2;          // ignore one-off co-occurrences — not a habit yet

  function enabled() { return !!S.getEditorSetting("suggestions"); }

  function libraryShortcuts() {
    return (S.get().shortcuts || []).filter((s) => s.enabled !== false && (s.triggers || []).length);
  }

  function keyOf(sc) { return String(sc.key || sc.name || (sc.triggers || [])[0] || ""); }

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

  function usesShortcut(textLc, sc) {
    return (sc.triggers || []).some((t) => containsTrigger(textLc, String(t || "").trim()));
  }

  // Shortcuts whose trigger appears in `text`.
  function presentIn(text) {
    const lc = String(text || "").toLowerCase();
    if (!lc.trim()) return [];
    return libraryShortcuts().filter((sc) => usesShortcut(lc, sc));
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
      if (!g.used && usesShortcut(textLc, sc)) g.used = true;
    }
    const missing = [...groups.entries()].filter(([, g]) => !g.used)
      .map(([cat, g]) => ({ cat, items: g.items }));
    const used = [...groups.entries()].filter(([, g]) => g.used)
      .map(([cat, g]) => ({ cat, items: g.items }));
    return { missing, used, total: groups.size };
  }

  function shuffled(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  // ── habit stats (mined server-side from all saved projects) ────────────────────
  // {counts, pairs:[[a,b,n]], follows:[[a,b,n]]} keyed by shortcut library key.
  function fetchStats() {
    if (!API || !API.suggestionStats) return Promise.resolve(null);
    return API.suggestionStats().catch(() => null);
  }

  function statMaps(stats) {
    const pair = new Map();        // key -> Map(partner -> n), symmetric
    const addPair = (a, b, n) => {
      if (!pair.has(a)) pair.set(a, new Map());
      pair.get(a).set(b, (pair.get(a).get(b) || 0) + n);
    };
    (stats.pairs || []).forEach(([a, b, n]) => { addPair(a, b, n); addPair(b, a, n); });
    const follow = new Map();      // a -> Map(b -> n), directional: shot a then shot b
    (stats.follows || []).forEach(([a, b, n]) => {
      if (!follow.has(a)) follow.set(a, new Map());
      follow.get(a).set(b, (follow.get(a).get(b) || 0) + n);
    });
    return { pair, follow };
  }

  // Rank candidates by habit strength against a set of context keys. Returns
  // [{sc, n, withKey}] — n = total co-occurrences, withKey = strongest partner.
  function rankByHabit(map, contextKeys, excludeKeys) {
    const byKey = new Map(libraryShortcuts().map((sc) => [keyOf(sc), sc]));
    const score = new Map();       // candidate key -> { n, best, bestN }
    for (const ctx of contextKeys) {
      const partners = map.get(ctx);
      if (!partners) continue;
      for (const [cand, n] of partners) {
        if (excludeKeys.has(cand) || !byKey.has(cand)) continue;
        const cur = score.get(cand) || { n: 0, best: ctx, bestN: 0 };
        cur.n += n;
        if (n > cur.bestN) { cur.best = ctx; cur.bestN = n; }
        score.set(cand, cur);
      }
    }
    return [...score.entries()]
      .filter(([, v]) => v.n >= MIN_HABIT_N)
      .sort((a, b) => b[1].n - a[1].n)
      .slice(0, MAX_HABIT)
      .map(([cand, v]) => ({ sc: byKey.get(cand), n: v.n, withSc: byKey.get(v.best) }));
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

  function bulb(ta, opts) {
    opts = opts || {};
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
        : "Ideas — pairings, next-shot habits, and category browse";
    }

    function position() {
      if (!pop) return;
      const r = btn.getBoundingClientRect();
      pop.style.top = (r.bottom + 4) + "px";
      pop.style.left = Math.max(8, Math.min(r.left, window.innerWidth - 340)) + "px";
    }

    function chipRow(sc, why) {
      const trig = String((sc.triggers || [])[0] || "").trim();
      if (!trig) return null;
      const row = el("div", "sugg-item");
      row.append(el("span", "ac-trigger", trig));
      const rep = (sc.replacements || [])[0];
      if (rep) row.append(el("span", "ac-prompt", rep));
      if (why) row.append(el("span", "sugg-why", why));
      else if (sc.sub_category) row.append(el("span", "ac-cat", sc.sub_category));
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

    function secHead(box, label) {
      box.append(el("div", "sugg-sec-head", label));
    }

    function trigOf(sc) { return String((sc.triggers || [])[0] || "").trim(); }

    function openPop() {
      closePop();
      document.querySelectorAll(".sugg-pop").forEach((p) => p.remove());
      // Never open off an invisible/detached bulb — the popover would strand at (0,0).
      const br = btn.getBoundingClientRect();
      if (!btn.isConnected || (!br.width && !br.height)) return;

      const { missing, used } = analyze(ta.value);
      const present = presentIn(ta.value);
      const presentKeys = new Set(present.map(keyOf));
      pop = el("div", "sugg-pop");

      const head = el("div", "sugg-head");
      head.append(el("span", "sugg-title", "Ideas"));
      const reroll = el("button", "btn ghost tiny", "⟳");
      reroll.title = "Shuffle the samples";
      reroll.onclick = (ev) => { ev.stopPropagation(); fillBrowse(); };
      head.append(reroll);
      pop.append(head);

      // ── habits (async): "you've paired these before" + "your usual next shot" ──
      const habits = el("div", "sugg-habits");
      pop.append(habits);
      fetchStats().then((stats) => {
        if (!pop || !stats) return;
        const { pair, follow } = statMaps(stats);
        // Pairings: shortcuts you historically put in the SAME shot as what's here.
        if (presentKeys.size) {
          const ideas = rankByHabit(pair, presentKeys, presentKeys);
          if (ideas.length) {
            secHead(habits, "You've paired these before");
            for (const i of ideas) {
              const row = chipRow(i.sc, `with «${trigOf(i.withSc)}» ×${i.n}`);
              if (row) habits.append(row);
            }
          }
        }
        // Next shot: what usually comes after the PREVIOUS shot's shortcuts.
        const prevText = typeof opts.prevText === "function" ? (opts.prevText() || "") : "";
        const prevKeys = new Set(presentIn(prevText).map(keyOf));
        if (prevKeys.size) {
          const ideas = rankByHabit(follow, prevKeys, presentKeys);
          if (ideas.length) {
            secHead(habits, "Your usual next shot");
            for (const i of ideas) {
              const row = chipRow(i.sc, `after «${trigOf(i.withSc)}» ×${i.n}`);
              if (row) habits.append(row);
            }
          }
        }
        position();
      });

      const list = el("div", "sugg-list");
      pop.append(list);

      // ── browse (sync): continue current categories + jump into missing ones ────
      const expanded = new Set();
      function fillBrowse() {
        list.replaceChildren();
        // Continue: more from categories this prompt already uses (minus what's in it).
        const contCats = used
          .map((u) => ({ cat: u.cat, items: u.items.filter((sc) => !presentKeys.has(keyOf(sc))) }))
          .filter((u) => u.items.length);
        if (contCats.length) {
          secHead(list, "Continue the thread");
          for (const u of shuffled(contCats).slice(0, 3)) {
            const headRow = el("div", "sugg-cat-head");
            headRow.append(el("span", "sugg-cat-name", "more " + u.cat));
            list.append(headRow);
            for (const sc of shuffled(u.items).slice(0, 2)) {
              const row = chipRow(sc);
              if (row) list.append(row);
            }
          }
        }
        // Jump: whole categories this prompt doesn't touch, with counts and browse.
        if (missing.length) {
          secHead(list, "New directions");
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
                fillBrowse();
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
        if (!contCats.length && !missing.length) {
          list.append(el("div", "sugg-empty",
            "Every shortcut category already appears here — nothing left to browse."));
        }
      }
      fillBrowse();

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
