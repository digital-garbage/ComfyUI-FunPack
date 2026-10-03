// What a prompt could still use, from the shortcut library and the person's own habits. Pure. A shortcut is keyed by its name.
const usable = (library) => library.filter((s) => s.enabled !== false && (s.triggers || []).length);
const isWord = (ch) => /[a-z0-9_]/.test(ch || "");

function hasTrigger(lc, trigger) {              // as a whole token: word boundaries on both sides, spaces allowed inside
  const t = String(trigger || "").trim().toLowerCase();
  for (let at = t ? lc.indexOf(t) : -1; at >= 0; at = lc.indexOf(t, at + 1)) if (!isWord(lc[at - 1]) && !isWord(lc[at + t.length])) return true;
  return false;
}
const uses = (lc, sc) => sc.triggers.some((t) => hasTrigger(lc, t));

/** The shortcuts whose trigger is in `text`. */
export const presentIn = (library, text) => { const lc = String(text || "").toLowerCase(); return lc.trim() ? usable(library).filter((s) => uses(lc, s)) : []; };

/** Library grouped by category: `used` have a trigger in the text, `missing` have none. */
export function analyze(library, text) {
  const lc = String(text || "").toLowerCase(), groups = new Map();
  for (const sc of usable(library)) {
    const cat = sc.category || "other";
    if (!groups.has(cat)) groups.set(cat, { cat, used: false, items: [] });
    const g = groups.get(cat);
    g.items.push(sc);
    g.used = g.used || uses(lc, sc);
  }
  const all = [...groups.values()];
  return { used: all.filter((g) => g.used), missing: all.filter((g) => !g.used) };
}

/** Habit partners of the context shortcuts: `edges` is [[a, b, n]] (stats.pairs is symmetric once read both ways).
 *  -> [{ sc, n, with }] strongest first; one-offs (n < 2) are not a habit yet. */
export function byHabit(library, edges, context, exclude, { both = false, limit = 4 } = {}) {
  const byName = new Map(usable(library).map((s) => [s.name, s])), score = new Map();
  for (const [a, b, n] of edges) {
    for (const [from, to] of both ? [[a, b], [b, a]] : [[a, b]]) {
      if (!context.has(from) || exclude.has(to) || !byName.has(to)) continue;
      const cur = score.get(to) || { n: 0, with: from, best: 0 };
      cur.n += n;
      if (n > cur.best) { cur.best = n; cur.with = from; }
      score.set(to, cur);
    }
  }
  return [...score].filter(([, v]) => v.n >= 2).sort((x, y) => y[1].n - x[1].n).slice(0, limit).map(([name, v]) => ({ sc: byName.get(name), n: v.n, with: byName.get(v.with) }));
}

export const shuffled = (list, random = Math.random) => {
  const a = list.slice();
  for (let i = a.length - 1; i > 0; i -= 1) { const j = Math.floor(random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
};
