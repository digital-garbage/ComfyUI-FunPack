// Keyboard shortcuts as a service: one listener, features bind combos like "mod+z" or "shift+mod+z".
// Typing in a field is never intercepted. A binding that returns false lets the key through.
export function createKeys(doc = document) {
  const table = new Map();
  const combo = (e) => [e.shiftKey && "shift", (e.metaKey || e.ctrlKey) && "mod", e.key.toLowerCase()].filter(Boolean).join("+");
  doc.addEventListener("keydown", (e) => {
    const t = e.target;
    if (t && (t.isContentEditable || /^(input|textarea|select)$/i.test(t.tagName))) return;
    const fn = table.get(combo(e));
    if (fn && fn(e) !== false) e.preventDefault();
  });
  return { bind(key, fn) { table.set(key, fn); return () => table.get(key) === fn && table.delete(key); } };
}
