// The picker wheel: press the middle mouse button anywhere and every offered action fans out under the pointer; release on one to run it.
import { composer as c } from "../../composer/composer.js";
import { offer } from "../../shell/actions.js";

// Per browser, not per project: the wheel is a habit of the hand. enabled null = everything offered.
const KEY = "funpack_wheel";
export function readPrefs() {
  try { const p = JSON.parse(localStorage.getItem(KEY)) || {}; return { shape: p.shape === "half" ? "half" : "full", side: p.side === "left" ? "left" : "right", enabled: Array.isArray(p.enabled) ? p.enabled : null }; }
  catch { return { shape: "full", side: "right", enabled: null }; }
}
const writePrefs = (p) => { try { localStorage.setItem(KEY, JSON.stringify(p)); } catch { /* private window: lasts until reload */ } };

export default {
  id: "wheel",
  mount: "menubar.menus",
  needs: ["project"],
  setup({ app }) {
    let wheel = null;
    const offs = [
      offer(app, { id: "undo", label: "Undo", icon: "↶", run: () => app.project.undo() }),
      offer(app, { id: "redo", label: "Redo", icon: "↷", run: () => app.project.redo() }),
      offer(app, { id: "models", label: "Models & Pipeline", icon: "▤", when: () => Boolean(app.openSettings), run: () => app.openSettings("models") }),
      offer(app, { id: "engine", label: "Engine settings", icon: "⚙", when: () => Boolean(app.openSettings), run: () => app.openSettings("engine") }),
    ];
    const onDown = (e) => {
      if (e.button !== 1 || wheel || document.querySelector('[role="dialog"]')) return;       // nothing behind a dialog
      e.preventDefault();
      const prefs = readPrefs(), on = (a) => !prefs.enabled || prefs.enabled.includes(a.id);
      const items = (app.actions || []).filter((a) => on(a) && (!a.when || a.when())).slice(0, 12).map((a) => ({ id: a.id, label: a.label, icon: a.icon }));
      if (!items.length) return;
      const shared = { items, onClose: () => { wheel = null; },
        onPick: (item) => { const a = (app.actions || []).find((x) => x.id === item.id); if (a) Promise.resolve().then(() => a.run()).catch((err) => c.toast.warn({ text: err.message })); } };
      wheel = prefs.shape === "half" ? c.wheel.half({ ...shared, edge: prefs.side, at: e.clientY }) : c.wheel.picker({ ...shared, x: e.clientX, y: e.clientY });
    };
    app.editorSettings.push(() => {
      const prefs = readPrefs(), set = (patch) => writePrefs({ ...readPrefs(), ...patch });
      const offered = (app.actions || []).map((a) => ({ value: a.id, label: `${a.icon || ""} ${a.label}`.trim() }));
      return c.region.stack({ gap: "sm", children: [c.label.section({ text: "Picker wheel" }),
        c.hint.default({ text: "Hold the middle mouse button anywhere, flick toward an action and let go." }),
        c.segmented.md({ label: "Shape", value: prefs.shape, onChange: (v) => set({ shape: v }), options: [{ value: "full", label: "Full wheel" }, { value: "half", label: "Half wheel" }] }),
        c.segmented.md({ label: "A half wheel opens", value: prefs.side, onChange: (v) => set({ side: v }), options: [{ value: "left", label: "Left" }, { value: "right", label: "Right" }] }),
        c.checklist.default({ label: "On the wheel", items: offered, values: prefs.enabled || offered.map((o) => o.value), onChange: (v) => set({ enabled: v }) })] });
    });
    document.addEventListener("mousedown", onDown, true);
    return () => { document.removeEventListener("mousedown", onDown, true); if (wheel) wheel.close && wheel.close(); offs.forEach((f) => f()); };
  },
};
