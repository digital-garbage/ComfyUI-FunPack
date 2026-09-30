// "Reactive focus": before Generate, list what the camera could aim at in each [Shot N]
// (step 1) and which view could open each later shot (step 2); let the person choose, and
// remember it. Each step appears only when its feature is on and something is undecided.
(function () {
  const el = (typeof window !== "undefined" && window.dom) ? window.dom.el : null;
  const MOVE_MODES = [["auto", "Auto"], ["move", "Move"], ["hold", "Hold"], ["none", "No move"]];
  const VIEW_MODES = [["auto", "Auto"], ["none", "No view"]];

  // Only shots the person has not decided yet (choices are keyed by the shot's own text).
  function pending(scenes, saved, field, usable) {
    field = field || "shots";
    usable = usable || ((s) => !s.already && s.candidates.length);
    return scenes.map((sc) => ({ ...sc, [field]: (sc[field] || []).filter((s) => usable(s) && !saved[s.key]) }))
      .filter((sc) => sc[field].length);
  }

  // One page of rows. -> {nodes, collect()}; collect() -> {choices, decisions}.
  function movesPage(scenes) {
    const state = new Map();
    const nodes = [];
    scenes.forEach((sc) => {
      nodes.push(el("div", "ov-label", `Scene ${sc.index + 1} — ${sc.preview}`));
      sc.shots.forEach((s) => {
        const st = { mode: "auto", lemma: null };
        state.set(s.key, st);
        nodes.push(row(`Shot ${s.shot}`, s.candidates.slice(0, 6).map((c) => [c.lemma, c.text]), st,
          "lemma", s.auto_lemma, MOVE_MODES));
      });
    });
    return {
      nodes,
      collect() {
        const choices = {}, decisions = [];
        scenes.forEach((sc) => sc.shots.forEach((s) => {
          const st = state.get(s.key);
          choices[s.key] = { mode: st.mode, lemma: st.mode === "none" ? null : (st.lemma || s.auto_lemma) };
          decisions.push({ auto: s.auto_lemma, picked: choices[s.key].lemma, mode: st.mode });
        }));
        return { choices, decisions };
      },
    };
  }

  function viewsPage(scenes) {
    const state = new Map();
    const nodes = [];
    scenes.forEach((sc) => {
      nodes.push(el("div", "ov-label", `Scene ${sc.index + 1} — views`));
      sc.views.forEach((s) => {
        const st = { mode: "auto", view: null };
        state.set(s.key, st);
        nodes.push(row(`Shot ${s.shot}`, s.candidates.map((c) => [c.view, c.view]), st, "view", s.auto, VIEW_MODES));
      });
    });
    return {
      nodes,
      collect() {
        const choices = {}, decisions = [];
        scenes.forEach((sc) => sc.views.forEach((s) => {
          const st = state.get(s.key);
          if (st.mode === "none") choices[s.key] = { mode: "none" };
          else if (st.view) choices[s.key] = { mode: "pick", view: st.view };
          else choices[s.key] = { mode: "auto" };
          // Accepting "Auto" is not a pick: the draw may land on any allowed view.
          decisions.push({ auto: s.auto, picked: st.mode === "none" ? null : st.view, traits: s.traits });
        }));
        return { choices, decisions };
      },
    };
  }

  // A row: label, candidate chips (click to pick, click again to unpick), mode select.
  function row(label, options, st, field, autoValue, modes) {
    const r = el("div", "ov-field full");
    r.appendChild(el("span", "ov-label", label));
    const chips = el("div", "ov-checklist");
    chips.style.cssText = "flex-direction:row;flex-wrap:wrap;gap:6px";
    const btns = options.map(([value, text]) => {
      const b = el("button", "btn", text);
      b.type = "button";
      b.onclick = () => { st[field] = st[field] === value ? null : value; paint(); };
      chips.appendChild(b);
      return [b, value];
    });
    const sel = document.createElement("select");
    sel.className = "ov-input";
    modes.forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; sel.appendChild(o); });
    sel.onchange = () => { st.mode = sel.value; paint(); };
    function paint() {
      btns.forEach(([b, v]) => b.classList.toggle("primary", (st[field] || autoValue) === v));
      chips.style.opacity = st.mode === "none" ? "0.4" : "";
    }
    paint();
    r.append(chips, sel);
    return r;
  }

  // steps: [{title, subtitle, page}]. -> Promise<[{choices, decisions}] | null> (null = cancelled)
  function ask(steps) {
    return new Promise((resolve) => {
      let done = false, at = 0;
      const { body, foot, close } = window.OverlayUI.openModal({
        title: "Reactive focus",
        subtitle: "",
        widthClass: "ov-modal-wide",
        onClose: () => { if (!done) { done = true; resolve(null); } },
      });
      const show = () => {
        body.replaceChildren(el("div", "ov-label", steps[at].title), ...steps[at].page.nodes);
        foot.replaceChildren(...buttons());
      };
      const finish = (useChoices) => {
        done = true;
        close();
        resolve(steps.map((s) => (useChoices ? s.page.collect() : { choices: {}, decisions: [] })));
      };
      const buttons = () => {
        const last = at === steps.length - 1;
        const cancel = el("button", "btn", "Cancel");
        const skip = el("button", "btn", "Skip — let it choose");
        const go = el("button", "btn primary", last ? "Generate" : "Next: " + steps[at + 1].title);
        cancel.onclick = () => close();
        skip.onclick = () => finish(false);
        go.onclick = () => { if (last) finish(true); else { at++; show(); } };
        const out = [cancel, skip];
        if (at) { const back = el("button", "btn", "Back"); back.onclick = () => { at--; show(); }; out.push(back); }
        out.push(go);
        return out;
      };
      show();
    });
  }

  // Called by Store before it queues a run. -> false to abort the run.
  async function review(project, sceneIds) {
    const { rf } = window.StudioSettings.read(project);
    const chance = (v, d) => (v == null ? d : Number(v));
    const wantMoves = !!(rf.camera_moves && rf.reactive_focus && chance(rf.camera_moves_chance, 0.7) > 0);
    const wantViews = !!(rf.shot_views && rf.reactive_focus && chance(rf.shot_view_chance, 0.4) > 0);
    if (!wantMoves && !wantViews) return true;
    let scenes;
    try {
      scenes = (await window.MovieEditorAPI.focusOptions(project.id, sceneIds)).scenes || [];
    } catch (e) {
      console.warn("[FunPack] reactive focus: could not list targets —", e.message);
      window.Store.get().notice = `Reactive focus was skipped: ${e.message}`;
      return true;
    }
    const objOr = (v) => (v && typeof v === "object" ? v : {});
    const savedMoves = objOr(rf.focus_choices), savedViews = objOr(rf.view_choices);
    const steps = [];
    if (wantMoves) {
      const todo = pending(scenes, savedMoves);
      if (todo.length) steps.push({ kind: "moves", title: "Camera focus", page: movesPage(todo) });
    }
    if (wantViews) {
      const todo = pending(scenes, savedViews, "views", (s) => !s.already && s.candidates.length);
      if (todo.length) steps.push({ kind: "views", title: "Views", page: viewsPage(todo) });
    }
    if (!steps.length) return true;
    const res = await ask(steps);
    if (!res) return false;
    const patch = {}, learn = { decisions: [], views: [] };
    steps.forEach((s, i) => {
      if (!res[i].decisions.length) return;
      if (s.kind === "moves") { patch.focus_choices = { ...savedMoves, ...res[i].choices }; learn.decisions = res[i].decisions; }
      else { patch.view_choices = { ...savedViews, ...res[i].choices }; learn.views = res[i].decisions; }
    });
    if (Object.keys(patch).length) {
      window.StudioSettings.patchRefiner(patch, true);
      window.MovieEditorAPI.focusLearn(learn.decisions, learn.views).catch((e) => console.warn("[FunPack] focus memory:", e.message));
    }
    return true;
  }

  if (typeof window !== "undefined") window.ReactiveFocus = { review, pending };
  if (typeof module !== "undefined") module.exports = { pending };
})();
