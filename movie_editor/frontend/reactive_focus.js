// "Reactive focus": before Generate, list what the camera could aim at in each [Shot N],
// let the person choose (target, hold vs move, or no move), and remember it.
(function () {
  const el = (typeof window !== "undefined" && window.dom) ? window.dom.el : null;
  const MODES = [["auto", "Auto"], ["move", "Move"], ["hold", "Hold"], ["none", "No move"]];

  // Only shots the person has not decided yet (choices are keyed by the shot's own text).
  function pending(scenes, saved) {
    return scenes.map((sc) => ({ ...sc, shots: sc.shots.filter((s) => !s.already && s.candidates.length && !saved[s.key]) }))
      .filter((sc) => sc.shots.length);
  }

  // -> Promise<{choices, decisions} | null>  (null = cancelled)
  function ask(scenes) {
    return new Promise((resolve) => {
      const O = window.OverlayUI;
      let done = false;
      const { body, foot, close } = O.openModal({
        title: "Reactive focus",
        subtitle: "Where should the camera look in each shot? Your picks are remembered and steer future suggestions.",
        widthClass: "ov-modal-wide",
        onClose: () => { if (!done) { done = true; resolve(null); } },
      });
      const state = new Map();   // shot key -> {mode, lemma|null}
      scenes.forEach((sc) => {
        body.appendChild(el("div", "ov-label", `Scene ${sc.index + 1} — ${sc.preview}`));
        sc.shots.forEach((s) => {
          const st = { mode: "auto", lemma: null };
          state.set(s.key, st);
          const row = el("div", "ov-field full");
          row.appendChild(el("span", "ov-label", `Shot ${s.shot}`));
          const chips = el("div", "ov-checklist");
          const chipBtns = s.candidates.slice(0, 6).map((c) => {
            const b = el("button", "btn", c.text);
            b.type = "button";
            b.onclick = () => { st.lemma = st.lemma === c.lemma ? null : c.lemma; paint(); };
            return [b, c];
          });
          chipBtns.forEach(([b]) => chips.appendChild(b));
          const sel = document.createElement("select");
          sel.className = "ov-input";
          MODES.forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; sel.appendChild(o); });
          sel.onchange = () => { st.mode = sel.value; paint(); };
          function paint() {
            chipBtns.forEach(([b, c]) => b.classList.toggle("primary", (st.lemma || s.auto_lemma) === c.lemma));
            chips.style.opacity = st.mode === "none" ? "0.4" : "";
          }
          paint();
          row.append(chips, sel);
          body.appendChild(row);
        });
      });
      const finish = (useChoices) => {
        const choices = {}, decisions = [];
        if (useChoices) {
          scenes.forEach((sc) => sc.shots.forEach((s) => {
            const st = state.get(s.key);
            choices[s.key] = { mode: st.mode, lemma: st.mode === "none" ? null : (st.lemma || s.auto_lemma) };
            decisions.push({ auto: s.auto_lemma, picked: choices[s.key].lemma, mode: st.mode });
          }));
        }
        done = true;
        close();
        resolve({ choices, decisions });
      };
      const cancel = el("button", "btn", "Cancel");
      const skip = el("button", "btn", "Skip — let it choose");
      const go = el("button", "btn primary", "Generate");
      cancel.onclick = () => close();
      skip.onclick = () => finish(false);
      go.onclick = () => finish(true);
      foot.append(cancel, skip, go);
    });
  }

  // Called by Store before it queues a run. -> false to abort the run.
  async function review(project, sceneIds) {
    const { rf } = window.StudioSettings.read(project);
    if (!rf.camera_moves || !rf.reactive_focus) return true;
    let scenes;
    try {
      scenes = (await window.MovieEditorAPI.focusOptions(project.id, sceneIds)).scenes || [];
    } catch (e) {
      console.warn("[FunPack] reactive focus: could not list targets —", e.message);
      return true;
    }
    const saved = (rf.focus_choices && typeof rf.focus_choices === "object") ? rf.focus_choices : {};
    const todo = pending(scenes, saved);
    if (!todo.length) return true;
    const res = await ask(todo);
    if (!res) return false;
    if (res.decisions.length) {
      window.StudioSettings.patchRefiner({ focus_choices: { ...saved, ...res.choices } }, true);
      window.MovieEditorAPI.focusLearn(res.decisions).catch((e) => console.warn("[FunPack] focus memory:", e.message));
    }
    return true;
  }

  if (typeof window !== "undefined") window.ReactiveFocus = { review, pending };
  if (typeof module !== "undefined") module.exports = { pending };
})();
