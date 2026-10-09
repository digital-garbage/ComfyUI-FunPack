// What a prompt could still use, from the shortcut library and the person's own habits. Pure. A shortcut is keyed by its name.
const usable = (library) => library.filter((s) => s.enabled !== false && (s.triggers || []).length);
// The expander's own rule (core/shortcuts.py _trigger_pattern): no letter, digit, _, ' or - on either side; any run of space between words.
function hasTrigger(lc, trigger) {
  const words = String(trigger || "").trim().split(/\s+/).filter(Boolean).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return words.length > 0 && new RegExp(`(?<![\\p{L}\\p{N}_'’-])${words.join("\\s+")}(?![\\p{L}\\p{N}_'’-])`, "iu").test(lc);
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

/** A scene from the person's shortcuts: the idea's own triggers kept, plus up to `add` more, one per category the text
 *  does not use yet (an uncategorised shortcut is its own), drawn at random. Ratings set the odds: `stats.scores` and
 *  `stats.rated_pairs` (beside what is already in) multiply a shortcut's chance (/suggestion_stats). -> { text, added } */
export function compose(library, stats = {}, idea = "", { add = 3, random = Math.random } = {}) {
  const cat = (s) => s.category || `\0${s.name}`, key = (t) => String(t).trim().toLowerCase().replace(/\s+/g, " ");
  const owners = new Map();        // a trigger two shortcuts share fires whichever the expander meets first: never offered
  for (const s of usable(library)) for (const t of s.triggers) if (key(t)) owners.set(key(t), (owners.get(key(t)) || 0) + 1);
  const trigger = (s) => s.triggers.find((t) => key(t) && owners.get(key(t)) === 1);
  const here = presentIn(library, idea), cats = new Set(here.map(cat)), taken = new Set(here.map((s) => s.name));
  const scores = stats.scores || {}, near = new Map(), added = [];
  const meet = (name) => { for (const [a, b, n] of stats.rated_pairs || []) for (const [x, y] of [[a, b], [b, a]]) if (x === name) near.set(y, (near.get(y) ?? 1) * n); };
  here.forEach((s) => meet(s.name));
  const pool = usable(library).filter((s) => (s.replacements || []).some((r) => String(r).trim()) && trigger(s));
  while (added.length < add) {
    // Below 1 (disliked), a shortcut sits a draw out that often, even with no rival; above, it outweighs the rest.
    const open = pool.filter((s) => !taken.has(s.name) && !cats.has(cat(s))).map((s) => ({ s, w: (scores[s.name] ?? 1) * (near.get(s.name) ?? 1) }))
      .filter((x) => x.w >= 1 || random() < x.w).map((x) => ({ s: x.s, w: Math.max(x.w, 1) }));
    if (!open.length) break;
    let r = random() * open.reduce((t, x) => t + x.w, 0);
    const { s } = open.find((x) => (r -= x.w) < 0) || open[open.length - 1];
    added.push(s); taken.add(s.name); cats.add(cat(s)); meet(s.name);
  }
  // Commas between: two triggers side by side must not read as a third ("golden" + "hour" = "golden hour").
  return { text: [String(idea || "").trim(), ...added.map((s) => trigger(s).trim())].filter(Boolean).join(", "), added };
}
