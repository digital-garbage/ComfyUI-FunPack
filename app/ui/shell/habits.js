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

/** A scene from the person's shortcuts, the best-rated first: the idea's own triggers kept, plus up to `add` more, one per
 *  category the text does not use yet, drawn at random by weight. A shortcut rated down overall, or rated down alongside
 *  what is already here, is never drawn. `stats` is /suggestion_stats. -> { text, added: [shortcut] } */
export function compose(library, stats = {}, idea = "", { add = 3, random = Math.random } = {}) {
  const here = presentIn(library, idea), names = new Set(here.map((s) => s.name)), cats = new Set(here.map((s) => s.category || "other"));
  const scores = stats.scores || {}, counts = stats.counts || {}, near = new Map();
  for (const [a, b, n] of stats.rated_pairs || []) for (const [x, y] of [[a, b], [b, a]]) if (names.has(x)) near.set(y, (near.get(y) || 0) + n);
  const pool = usable(library).filter((s) => !names.has(s.name) && (s.replacements || []).length && (scores[s.name] || 0) >= 0 && (near.get(s.name) || 0) >= 0)
    .map((s) => ({ s, w: 1 + 2 * ((scores[s.name] || 0) + (near.get(s.name) || 0)) + (counts[s.name] || 0) / 4 }));
  const added = [];
  while (added.length < add) {
    const open = pool.filter((x) => !cats.has(x.s.category || "other"));
    if (!open.length) break;
    let r = random() * open.reduce((t, x) => t + x.w, 0);
    const pick = open.find((x) => (r -= x.w) < 0) || open[open.length - 1];
    added.push(pick.s);
    cats.add(pick.s.category || "other");
  }
  const trigger = (s) => s.triggers.find((t) => String(t).trim()).trim();
  return { text: [String(idea || "").trim(), ...added.map(trigger)].filter(Boolean).join(" "), added };
}
