// Captions as tags, the same rules as the server's modules/training/lora/captions.py.

export const splitTags = (text) => {
  const seen = new Set(), out = [];
  for (const part of String(text ?? "").replace(/\n/g, ",").split(",")) {
    const tag = part.split(/\s+/).filter(Boolean).join(" ");
    if (tag && !seen.has(tag.toLowerCase())) { seen.add(tag.toLowerCase()); out.push(tag); }
  }
  return out;
};
export const joinTags = (tags) => splitTags((tags || []).join(", ")).join(", ").slice(0, 4000);

const same = (a, b) => String(a).split(/\s+/).filter(Boolean).join(" ").toLowerCase() === String(b).split(/\s+/).filter(Boolean).join(" ").toLowerCase();
/** The comma separated pieces of a caption as typed: trimmed, no empties, nothing dropped or reworded. */
export const parts = (text) => String(text ?? "").split(",").map((p) => p.trim()).filter(Boolean);

/** The caption with these tags added at its end after a comma, skipping any it has; the caption itself stays as typed. */
export function addTags(caption, tags) {
  const have = parts(caption);
  const fresh = splitTags((tags || []).join(", ")).filter((t) => !have.some((h) => same(t, h)));
  const text = String(caption ?? "").trimEnd().replace(/,+$/, "").trimEnd();
  return (text ? [text, ...fresh] : fresh).join(", ").slice(0, 4000);
}

/** Every tag with how many captions hold it, most common first. */
export function frequency(items) {
  const counts = new Map();
  for (const item of items) for (const tag of splitTags(item.caption)) {
    const key = tag.toLowerCase(), have = counts.get(key);
    if (have) have.count += 1; else counts.set(key, { tag, count: 1 });
  }
  return [...counts.values()].sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
}

/** Change captions in place; returns how many changed. op: add, prepend, remove, replace. `ids` (a Set) limits it. */
export function bulk(items, op, tag, to, ids) {
  tag = String(tag ?? "").split(/\s+/).filter(Boolean).join(" ");
  if (!tag || !["add", "prepend", "remove", "replace"].includes(op)) return 0;
  to = String(to ?? "").split(/\s+/).filter(Boolean).join(" ");
  if (op === "replace" && !to) op = "remove";
  let changed = 0;
  for (const item of items) {
    if (ids && !ids.has(item.id)) continue;
    let tags = parts(item.caption);
    const has = tags.some((t) => same(t, tag));
    if (op === "add" && !has) tags.push(tag);
    else if (op === "prepend") tags = [tag, ...tags.filter((t) => !same(t, tag))];
    else if (op === "remove" && has) tags = tags.filter((t) => !same(t, tag));
    else if (op === "replace" && has) tags = tags.map((t) => (same(t, tag) ? to : t));
    else continue;
    const text = tags.join(", ").slice(0, 4000);
    if (text !== item.caption) { item.caption = text; changed += 1; }
  }
  return changed;
}
