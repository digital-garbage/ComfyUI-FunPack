// 💡 Shortcut ideas: a bulb beside the prompt box you are in. It never opens by itself; it glows while shortcut categories go unused,
// and its list mixes your own habits (what you paired before, what usually follows the previous shot) with browsing the library.
import { composer as c } from "../../composer/composer.js";
import { claimFor } from "../../composer/internals/zlayer.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";
import { analyze, byHabit, presentIn, shuffled } from "./ideas.js";

const BOXES = new Set(["Story", "Prompt"]);
const first = (sc) => String((sc.triggers || [])[0] || "").trim();

export default {
  id: "ideas",
  mount: "menubar.menus",
  needs: ["project", "api", "selection"],
  setup({ app }) {
    const p = app.project, on = () => p.pref("suggestions", true);
    let box = null, node = null, layer = null, pop = null, hold = 0;
    const closeBulb = () => { if (pop) pop.close(); if (node) { node.remove(); layer.release(); node = layer = box = null; } };
    const prevText = () => {                       // the previous shot's prompt, for "your usual next shot" (scene prompt only)
      const open = p.project; if (!open || box.getAttribute("aria-label") !== "Prompt") return "";
      const roots = open.scenes.filter((s) => isGenerative(s) && unitRoot(open, genUnitId(s)) === s), at = roots.findIndex((s) => s.id === app.selection.focus);
      return at > 0 ? roots[at - 1].text || "" : "";
    };
    const insert = (text) => {
      const v = box.value, s = box.selectionStart ?? v.length, e = box.selectionEnd ?? s, before = v.slice(0, s), after = v.slice(e);
      const ins = `${before && !/\s$/.test(before) ? " " : ""}${text}${/^(\s|[,;])/.test(after) ? "" : " "}`;
      box.value = before + ins + after; box.setSelectionRange(s + ins.length, s + ins.length);
      box.dispatchEvent(new Event("input", { bubbles: true })); box.focus();
    };

    async function openList() {
      if (pop) return pop.close();
      let library = [], stats = null;
      try { library = (await app.api.shortcuts()).shortcuts || []; } catch (err) { return c.toast.warn({ text: err.message }); }
      try { stats = await app.api.suggestionStats(); } catch { /* habits are optional */ }
      const text = box.value, { used, missing } = analyze(library, text), here = presentIn(library, text), hereNames = new Set(here.map((s) => s.name));
      const body = c.region.stack({ gap: "xs" });
      const row = (sc, why) => c.button.sm({ label: `${first(sc)}${(sc.replacements || [])[0] ? ` — ${sc.replacements[0].slice(0, 40)}` : ""}${why ? `  (${why})` : ""}`, tone: "ghost", onClick: () => { insert(first(sc)); pop && pop.close(); } });
      const draw = () => {
        const parts = [];
        if (stats && hereNames.size) {
          const pairs = byHabit(library, stats.pairs || [], hereNames, hereNames, { both: true });
          if (pairs.length) parts.push(c.label.section({ text: "You've paired these before" }), ...pairs.map((i) => row(i.sc, `with ${first(i.with)} ×${i.n}`)));
        }
        const prev = new Set(presentIn(library, prevText()).map((s) => s.name));
        if (stats && prev.size) {
          const next = byHabit(library, stats.follows || [], prev, hereNames);
          if (next.length) parts.push(c.label.section({ text: "Your usual next shot" }), ...next.map((i) => row(i.sc, `after ${first(i.with)} ×${i.n}`)));
        }
        const more = used.map((g) => ({ cat: g.cat, items: g.items.filter((s) => !hereNames.has(s.name)) })).filter((g) => g.items.length);
        if (more.length) parts.push(c.label.section({ text: "Continue the thread" }), ...shuffled(more).slice(0, 3).flatMap((g) => [c.hint.default({ text: `more ${g.cat}` }), ...shuffled(g.items).slice(0, 2).map((s) => row(s))]));
        if (missing.length) parts.push(c.label.section({ text: "New directions" }), ...missing.sort((a, b) => a.cat.localeCompare(b.cat)).flatMap((g) => [c.hint.default({ text: `${g.cat} (${g.items.length})` }), ...shuffled(g.items).slice(0, 3).map((s) => row(s))]));
        if (!parts.length) parts.push(c.hint.default({ text: "Every shortcut category already appears here — nothing left to browse." }));
        body.set([c.toolbar.default({ items: [c.label.section({ text: "Ideas" })], trailing: [c.button.sm({ label: "⟳", tone: "ghost", title: "Shuffle the samples", onClick: draw })] }), ...parts]);
      };
      draw();
      pop = c.popover.anchored({ anchor: node, body, side: "bottom", align: "end", onClose: () => { pop = null; } });
    }

    function show(target) {
      if (box === target || !on()) return;
      closeBulb();
      box = target;
      const button = c.button.sm({ label: "💡", tone: "ghost", title: "Shortcut ideas", onClick: openList });
      node = button.node;
      Object.assign(node.style, { position: "fixed" });
      document.body.append(node); layer = claimFor("popover", node);
      place();
    }
    function place() {
      if (!box || !node) return;
      const r = box.isConnected ? box.getBoundingClientRect() : null;
      if (!r || !r.width) return closeBulb();        // gone, or hidden with its panel
      Object.assign(node.style, { left: `${Math.max(4, r.right - 40)}px`, top: `${Math.max(4, r.top - 30)}px` });
    }
    const onFocus = (e) => { const t = e.target; if (t instanceof HTMLTextAreaElement && BOXES.has(t.getAttribute("aria-label"))) { clearTimeout(hold); show(t); } };
    const onBlur = (e) => { if (e.target === box) hold = setTimeout(() => { if (node && !node.contains(document.activeElement) && !pop) closeBulb(); }, 400); };
    const onMove = () => place();
    document.addEventListener("focusin", onFocus); document.addEventListener("focusout", onBlur);
    window.addEventListener("scroll", onMove, true); window.addEventListener("resize", onMove);

    app.editorSettings.push(() => c.region.stack({ gap: "sm", children: [
      c.toggle.default({ label: "Shortcut ideas", hint: "A 💡 beside prompt boxes lights up when shortcut categories in your library aren't used in the prompt. Click it for insertable ideas; it never pops up on its own.",
        checked: on(), onChange: (v) => { p.setPref("suggestions", v); if (!v) closeBulb(); } })] }));
    return () => { closeBulb(); clearTimeout(hold); document.removeEventListener("focusin", onFocus); document.removeEventListener("focusout", onBlur); window.removeEventListener("scroll", onMove, true); window.removeEventListener("resize", onMove); };
  },
};
