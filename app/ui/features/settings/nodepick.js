// A search box over every installed node. -> the chosen node class, or null.
import { composer as c } from "../../composer/composer.js";

export function pickNode(api, title) {
  return new Promise((resolve) => {
    const list = c.region.stack({ gap: "xs" }), note = c.hint.default({ text: "" });
    let timer = 0, ticket = 0, done = false;
    const finish = (v) => { if (!done) { done = true; resolve(v); } };
    const win = c.modal.generic({ title, size: "md", onClose: () => finish(null),
      body: c.region.stack({ gap: "sm", children: [c.input.md({ label: "Search", placeholder: "Search installed nodes by name or category…", onInput: (q) => { clearTimeout(timer); timer = setTimeout(() => run(q), 200); } }), note, list] }) });
    async function run(q) {
      const mine = ++ticket;
      let found;
      try { found = await api.searchNodes(q, 40); } catch (err) { return note.setText(`Could not search: ${err.message}`); }
      if (mine !== ticket) return;
      const nodes = found.nodes || [];
      list.set(nodes.map((n) => c.button.sm({ label: `${n.title || n.node}  ${n.category || ""}${n.outputs && n.outputs.length ? ` → ${n.outputs.join(", ")}` : ""}`, tone: "ghost", onClick: () => { finish(n.node); win.close("done"); } })));
      note.setText(found.total > nodes.length ? `${nodes.length} of ${found.total} shown: type more to narrow it.` : nodes.length ? "" : "Nothing matches.");
    }
    run("");
  });
}
