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
      nodes.push(el("div", "ov-label", `Scene ${sc.index + 1}`));
      sc.shots.forEach((s) => {
        const st = { mode: "auto", picks: [] };
        state.set(s.key, st);
        nodes.push(row(s, st, s.candidates.slice(0, 8).map((c) => [c.lemma, c.text]), s.auto_lemma, MOVE_MODES,
          "Click several, in order: the camera travels from the first to the last. Hold aims at the first only."));
      });
    });
    return {
      nodes,
      collect() {
        const choices = {}, decisions = [];
        scenes.forEach((sc) => sc.shots.forEach((s) => {
          const st = state.get(s.key);
          const lemmas = st.picks.length ? st.picks : [s.auto_lemma];
          choices[s.key] = st.mode === "none" ? { mode: "none", lemma: null }
            : { mode: st.mode, lemma: lemmas[0], lemmas };
          decisions.push({ auto: s.auto_lemma, picked: st.mode === "none" ? [] : lemmas, mode: st.mode });
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
        const st = { mode: "auto", picks: [] };
        state.set(s.key, st);
        nodes.push(row(s, st, s.candidates.map((c) => [c.view, c.view]), null, VIEW_MODES,
          "Click every view you are happy with: one of them is used."));
      });
    });
    return {
      nodes,
      collect() {
        const choices = {}, decisions = [];
        scenes.forEach((sc) => sc.views.forEach((s) => {
          const st = state.get(s.key);
          if (st.mode === "none") choices[s.key] = { mode: "none" };
          else if (st.picks.length) choices[s.key] = { mode: "pick", views: st.picks.slice() };
          else choices[s.key] = { mode: "auto" };
          // Accepting "Auto" is not a pick: the draw may land on any allowed view.
          decisions.push({ auto: s.auto, picked: st.mode === "none" ? [] : st.picks, traits: s.traits });
        }));
        return { choices, decisions };
      },
    };
  }

  // A row: what the shot says, candidate chips (click to add/remove; the number is the order),
  // mode select. `autoValue` is highlighted (dimmed) until the person picks something.
  function row(shot, st, options, autoValue, modes, hint) {
    const r = el("div", "ov-field full");
    r.appendChild(el("span", "ov-label", `Shot ${shot.shot}`));
    if (shot.raw) {
      const typed = el("div", "", shot.raw);
      typed.style.cssText = "font:12px ui-monospace,monospace;opacity:.85;margin:2px 0";
      r.appendChild(typed);
    }
    if (shot.text && shot.text !== shot.raw) {
      const said = el("div", "", shot.text);
      said.title = "Click to show all";
      said.style.cssText = "font-size:12px;opacity:.6;cursor:pointer;margin-bottom:4px;"
        + "display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden";
      said.onclick = () => { said.style.webkitLineClamp = said.style.webkitLineClamp === "unset" ? "2" : "unset"; };
      r.appendChild(said);
    }
    const chips = el("div", "ov-checklist");
    chips.style.cssText = "flex-direction:row;flex-wrap:wrap;gap:6px";
    const btns = options.map(([value, text]) => {
      const b = el("button", "btn", text);
      b.type = "button";
      b.onclick = () => {
        const at = st.picks.indexOf(value);
        if (at >= 0) st.picks.splice(at, 1); else st.picks.push(value);
        paint();
      };
      chips.appendChild(b);
      return [b, value, text];
    });
    const sel = document.createElement("select");
    sel.className = "ov-input";
    modes.forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; sel.appendChild(o); });
    sel.onchange = () => { st.mode = sel.value; paint(); };
    function paint() {
      btns.forEach(([b, v, text]) => {
        const at = st.picks.indexOf(v);
        const on = at >= 0 || (!st.picks.length && autoValue === v);
        b.classList.toggle("primary", on);
        b.textContent = at >= 0 && (st.picks.length > 1 || options.length > 0) ? `${at + 1} · ${text}` : text;
      });
      chips.style.opacity = st.mode === "none" ? "0.4" : "";
    }
    paint();
    const note = el("div", "", hint);
    note.style.cssText = "font-size:11px;opacity:.5;margin-top:4px";
    r.append(chips, sel, note);
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
        sticky: true,      // a stray click must not cancel the run; only × / Cancel / a step does
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

  // Why no shot was offered a view: the honest reason per shot.
  function whyNone(scenes, saved) {
    const all = scenes.flatMap((sc) => sc.views || []);
    if (!all.length) return "a view only opens shots 2 and later, so the prompt needs two or more [Shot N] blocks.";
    return all.map((s) => {
      if (s.already) return `shot ${s.shot} already states “${s.stated}”`;
      if (saved[s.key]) return `shot ${s.shot} is already decided (forget it in Settings ▸ Refinement & Taste)`;
      return `no view can show shot ${s.shot}`;
    }).join("; ") + ".";
  }

  // Called by Store before it queues a run. -> false to abort the run.
  async function review(project, sceneIds) {
    const { rf } = window.StudioSettings.read(project);
    const chance = (v, d) => (v == null ? d : Number(v));
    const wantMoves = !!(rf.camera_moves && rf.reactive_focus && chance(rf.camera_moves_chance, 0.7) > 0);
    const wantViews = !!(rf.shot_views && rf.reactive_focus && chance(rf.shot_view_chance, 0.4) > 0);
    if (!wantMoves && !wantViews) return true;
    let scenes, seen = {};
    try {
      seen = await window.MovieEditorAPI.focusOptions(project.id, sceneIds);
      scenes = seen.scenes || [];
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
    // A wanted step with nothing to ask says why, or "no views" looks like "broken".
    const whys = [];
    if (wantViews && !steps.some((x) => x.kind === "views")) whys.push("Views: " + whyNone(scenes, savedViews));
    if (!steps.length) {
      const why = !seen.with_shot_marks
        ? `no [Shot N] found in the ${seen.texts == null ? "" : seen.texts + " "}scene text(s) sent to generation.`
        : whys.join(" ") || "every shot is already decided or already has a camera move.";
      window.Store.get().notice = `Reactive focus: nothing to ask — ${why}`;
      return true;
    }
    if (whys.length) steps[steps.length - 1].page.nodes.push(el("div", "ov-label", whys.join(" ")));
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
