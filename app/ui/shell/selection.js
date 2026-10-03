// Which clips are selected: a set, one of them in focus (what the inspector shows), and an anchor for shift-ranges.
// Click picks one, cmd/ctrl toggles, shift extends from the anchor. Pure; the project says which scene is focused.
export function createSelection({ project }) {
  let ids = [];
  let anchor = null;
  const heard = new Set();
  const say = () => heard.forEach((fn) => { try { fn(); } catch { /* one listener must not stop the rest */ } });

  const live = () => new Set(project.scenes.map((s) => s.id));
  const heal = () => { const have = live(); ids = ids.filter((id) => have.has(id)); if (anchor && !have.has(anchor)) anchor = project.selectedId; };

  return {
    get ids() { heal(); return ids.length ? [...ids] : (project.selectedId ? [project.selectedId] : []); },
    has(id) { return this.ids.includes(id); },
    get focus() { return project.selectedId; },
    on: (fn) => { heard.add(fn); return () => heard.delete(fn); },
    /** `order`: scene ids as the timeline shows them, which is what a range means. */
    pick(id, { additive = false, range = false } = {}, order = project.scenes.map((s) => s.id)) {
      if (!project.scenes.some((s) => s.id === id)) return;
      const was = this.ids;
      if (range && anchor) {
        const a = order.indexOf(anchor), b = order.indexOf(id);
        ids = a < 0 || b < 0 ? [id] : order.slice(Math.min(a, b), Math.max(a, b) + 1);
      } else if (additive) {
        ids = was.includes(id) ? was.filter((x) => x !== id) : [...was, id];
        anchor = ids.includes(id) ? id : ids[ids.length - 1] || null;     // after un-picking, the last one still picked
      } else {
        ids = [id];
        anchor = id;
      }
      const focus = ids.includes(id) ? id : ids[ids.length - 1] || null;
      if (focus) project.select(focus);
      say();
    },
    clear() { ids = []; anchor = null; say(); },
    /** The project was replaced or restructured: drop what no longer exists. */
    heal() { heal(); say(); },
  };
}
