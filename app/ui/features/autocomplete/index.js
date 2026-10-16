// Prompt autocomplete: while typing in a prompt box, suggest the shortcut triggers that fit the word under the caret.
// Attaches to the prompt boxes by what they are called, when they are first focused; no other feature knows about it.
import { composer as c } from "../../composer/composer.js";
import { accept, suggestionsAt } from "./match.js";

const BOXES = new Set(["Story", "Prompt"]);          // aria-labels of the prompt fields

export default {
  id: "autocomplete",
  mount: "menubar.menus",
  needs: ["project", "api"],
  setup({ app }) {
    const p = app.project;
    let library = [], fetched = 0, span = null;
    const refresh = () => { if (Date.now() - fetched > 10000) { fetched = Date.now(); app.api.shortcuts().then((r) => { library = r.shortcuts || []; }).catch(() => {}); } };
    const live = new WeakMap();

    function attach(box) {
      if (live.has(box)) return;
      const source = () => {
        span = null;
        if (!p.pref("autocomplete", true) || box.selectionStart !== box.selectionEnd) return [];
        const found = suggestionsAt(library, box.value, box.selectionStart);
        if (!found) return [];
        span = found.span;
        return found.items.map((e) => ({ label: e.trigger, hint: (e.sc.replacements || [])[0] || [e.sc.category, e.sc.sub_category].filter(Boolean).join(" · "), trigger: e.trigger }));
      };
      const onPick = (item) => {
        if (!span) return;
        const done = accept(box.value, span, item.trigger);
        box.value = done.text; box.setSelectionRange(done.caret, done.caret);
        box.dispatchEvent(new Event("input", { bubbles: true }));
        box.focus();
      };
      live.set(box, c.autocomplete.default({ input: box, source, onPick }));
    }

    const onFocus = (e) => {
      const t = e.target;
      if (t instanceof HTMLTextAreaElement && BOXES.has(t.getAttribute("aria-label"))) { refresh(); attach(t); }
    };
    document.addEventListener("focusin", onFocus);

    app.editorSettings.push(() => c.region.stack({ gap: "sm", children: [
      c.toggle.default({ label: "Prompt autocomplete", hint: "Suggest matching shortcuts while you type in the Story and scene prompts. Shows the trigger and its prompt.",
        checked: p.pref("autocomplete", true), onChange: (v) => p.setPref("autocomplete", v) })] }));
    return () => document.removeEventListener("focusin", onFocus);
  },
};
