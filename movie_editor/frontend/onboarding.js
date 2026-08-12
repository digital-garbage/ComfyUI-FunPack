// First-run wizard, shared by the Cutting Room and Easy Gen.
//
// Shape is lifted from the macOS Setup Assistant: full screen, no window chrome, one
// vertically centred column per step, a glyph over a large headline over a short
// paragraph, controls in the middle, and a fixed action bar at the bottom whose primary
// button is always the way forward. Back is a chevron in the top-left; there are no step
// dots, because a linear flow that tells you how far you have to go invites you to bail.
//
// The steps a host runs are declared in HOSTS below, not branched on inside each step —
// Easy Gen has no timeline, so prompt splits are not a thing it can offer, and that
// belongs in one table rather than scattered through the screens.
(function () {
  const { el, clear } = window.dom;
  const S = () => window.Store;
  const API = () => window.MovieEditorAPI;
  const LS_DONE = "funpack_onboarded";

  let root = null;       // the full-screen overlay
  let stage = null;      // the animated content column
  let idx = 0;
  let steps = [];
  let ctx = null;        // per-run state: project, family, deps, generation type

  const isEasyGen = () => window.FunPackAppName === "Easy Gen";

  const HOSTS = {
    editor: {
      product: "Cutting Room",
      lead: "Multi-scene video on a real timeline.",
      steps: ["theme", "project", "prereqs", "gentype", "models", "extras", "done"],
      extras: ["links", "shortcuts", "splits"],
      newProject: (name) => S().newProject(name),
    },
    easy: {
      product: "Easy Gen",
      lead: "One prompt, one button, one video.",
      steps: ["theme", "project", "prereqs", "gentype", "models", "extras", "done"],
      extras: ["links", "shortcuts"],
      newProject: (name) => S().createProject(name),
    },
  };
  const host = () => (isEasyGen() ? HOSTS.easy : HOSTS.editor);

  // ── chrome ────────────────────────────────────────────────────────────────

  function glyph(paths, cls) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 48 48");
    svg.setAttribute("class", "oo-glyph" + (cls ? " " + cls : ""));
    svg.setAttribute("aria-hidden", "true");
    svg.innerHTML = paths;
    return svg;
  }

  const GLYPHS = {
    theme: '<circle cx="24" cy="24" r="15" fill="none" stroke="currentColor" stroke-width="2.4"/>'
      + '<path d="M24 9a15 15 0 0 0 0 30z" fill="currentColor"/>',
    project: '<rect x="7" y="11" width="34" height="26" rx="4" fill="none" stroke="currentColor" stroke-width="2.4"/>'
      + '<path d="M7 19h34" stroke="currentColor" stroke-width="2.4"/><circle cx="13" cy="15" r="1.6" fill="currentColor"/>',
    prereqs: '<path d="M24 6v22" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/>'
      + '<path d="M15 21l9 9 9-9" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
      + '<path d="M9 36h30" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/>',
    gentype: '<rect x="5" y="12" width="26" height="24" rx="4" fill="none" stroke="currentColor" stroke-width="2.4"/>'
      + '<path d="M31 21l12-7v20l-12-7z" fill="currentColor"/>',
    models: '<path d="M24 5 42 15v18L24 43 6 33V15z" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linejoin="round"/>'
      + '<path d="M6 15l18 10 18-10M24 25v18" fill="none" stroke="currentColor" stroke-width="2.4"/>',
    extras: '<circle cx="24" cy="24" r="5" fill="currentColor"/>'
      + '<path d="M24 4v8M24 36v8M4 24h8M36 24h8M10 10l6 6M32 32l6 6M38 10l-6 6M16 32l-6 6" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>',
    done: '<circle cx="24" cy="24" r="17" fill="none" stroke="currentColor" stroke-width="2.4"/>'
      + '<path d="M15 24.5l6.5 6.5L33 19" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>',
  };

  function head(icon, title, sub) {
    const h = el("div", "oo-head");
    if (icon) h.append(glyph(GLYPHS[icon], "oo-glyph-" + icon));
    h.append(el("h1", "oo-title", title));
    if (sub) h.append(el("p", "oo-sub", sub));
    return h;
  }

  // The action bar is fixed to the bottom of the screen on every step, so the primary
  // button never moves as the content above it changes height.
  function actions({ primary, onPrimary, secondary, onSecondary, disabled }) {
    const bar = el("div", "oo-actions");
    const btn = el("button", "oo-btn oo-btn-primary" + (disabled ? " disabled" : ""), primary || "Continue");
    btn.type = "button";
    if (!disabled) btn.onclick = onPrimary || next;
    bar.append(btn);
    if (secondary) {
      const s = el("button", "oo-btn oo-btn-quiet", secondary);
      s.type = "button";
      s.onclick = onSecondary || next;
      bar.append(s);
    }
    return bar;
  }

  function busy(text) {
    const b = el("div", "oo-busy");
    b.append(el("span", "oo-spinner"));
    b.append(el("span", null, text));
    return b;
  }

  // ── step registry ─────────────────────────────────────────────────────────
  // Each returns a DOM node; `enter` may be async and runs before the render.

  const STEPS = {
    // ── colour scheme ───────────────────────────────────────────────────────
    theme: {
      render() {
        const box = el("div", "oo-step");
        box.append(head("theme", "Choose your look",
          "You can change this any time in Settings ▸ Appearance."));
        const { grid } = window.AppearanceSettings.buildPicker();
        grid.classList.add("oo-theme-cards");
        box.append(grid);
        box.append(actions({}));
        return box;
      },
    },

    // ── project name + model family ─────────────────────────────────────────
    project: {
      async enter() {
        // Families come from the backend so the wizard can't drift from what the
        // pipeline actually supports, including the "not released yet" flags.
        try { ctx.deps = await API().pipelineDeps(null); } catch (_) { ctx.deps = null; }
      },
      render() {
        const box = el("div", "oo-step oo-step-wide");
        box.append(head("project", "Name it, and pick a model",
          "The model decides which nodes and which files this project needs, so it is asked first."));

        const nameRow = el("div", "oo-field");
        nameRow.append(el("label", "oo-label", "Project name"));
        const input = el("input", "oo-input");
        input.type = "text";
        input.value = ctx.name || (isEasyGen() ? "Untitled" : "Untitled montage");
        input.placeholder = "Project name";
        input.oninput = () => { ctx.name = input.value; };
        nameRow.append(input);
        box.append(nameRow);

        const families = (ctx.deps?.families || []).slice();
        const cards = el("div", "oo-cards");
        const paint = () => {
          [...cards.children].forEach((c) => c.classList.toggle("active", c.dataset.key === ctx.family));
          bar.replaceWith(bar = mkBar());
        };
        const card = (key, title, note, sub) => {
          const c = el("button", "oo-card");
          c.type = "button";
          c.dataset.key = key;
          const t = el("div", "oo-card-title", title);
          if (note) t.append(el("span", "oo-card-note", note));
          c.append(t);
          c.append(el("div", "oo-card-sub", sub || ""));
          c.onclick = () => { ctx.family = key; paint(); };
          return c;
        };
        families.forEach((f) => cards.append(
          card(f.key, f.label, f.released ? "" : "not released yet", f.summary)));
        cards.append(card("__own__", "My own pipeline", "",
          "Disable the built-in graph. Generation runs only the nodes you wire yourself in Models."));
        box.append(cards);

        let bar = mkBar();
        box.append(bar);
        function mkBar() {
          return actions({
            primary: "Continue",
            disabled: !ctx.family,
            onPrimary: createProject,
          });
        }
        if (ctx.family) paint();
        setTimeout(() => input.focus(), 60);
        return box;
      },
    },

    // ── missing node packs ──────────────────────────────────────────────────
    prereqs: {
      async enter() {
        ctx.deps = null;
        try { ctx.deps = await API().pipelineDeps(ctx.project?.id || null); } catch (_) {}
      },
      // Nothing to install and nothing to warn about — don't show a screen that only
      // says "all good" and asks you to press Continue.
      skip() {
        if (ctx.family === "__own__") return true;
        const d = ctx.deps;
        if (!d) return true;
        const r = d.readiness || {};
        return !d.needs_setup && !(d.missing_packs || []).length && !r.note && r.released !== false;
      },
      render() {
        const d = ctx.deps || {};
        const r = d.readiness || {};
        const packs = d.missing_packs || [];
        const box = el("div", "oo-step oo-step-wide");
        box.append(head("prereqs", packs.length ? "A few nodes are missing" : "About this model",
          packs.length
            ? "This model's pipeline needs custom node packs that aren't in your ComfyUI yet."
            : "Worth knowing before you download anything."));

        const panel = el("div", "oo-panel");
        if (r.note) panel.append(el("p", "oo-note", r.note));
        if (r.source_url) {
          const a = el("a", "oo-link", r.source_title || r.source_url);
          a.href = r.source_url; a.target = "_blank"; a.rel = "noopener";
          panel.append(a);
        }
        if (packs.length) {
          panel.append(el("div", "oo-list-label", "Node packs to install"));
          const ul = el("ul", "oo-list");
          packs.forEach((p) => {
            const li = el("li", null, p.title);
            if (p.missing_classes?.length) li.append(el("span", "oo-list-sub", p.missing_classes.join(", ")));
            ul.append(li);
          });
          panel.append(ul);
        }
        if ((r.models || []).length) {
          panel.append(el("div", "oo-list-label",
            r.released ? "Model files this pipeline uses" : "Model files to download once published"));
          const ul = el("ul", "oo-list");
          r.models.forEach((m) => {
            const li = el("li", null, m.label);
            li.append(el("span", "oo-list-sub", "models/" + m.folder + (m.hint ? " — " + m.hint : "")));
            ul.append(li);
          });
          panel.append(ul);
        }
        if (panel.children.length) box.append(panel);

        // Installing restarts ComfyUI, which reloads this page — the wizard resumes from
        // its saved position rather than starting over. See restore().
        let primary = "Continue", onPrimary = next, secondary = null, onSecondary = null;
        if (packs.length && d.needs_manager_install) {
          primary = "Install ComfyUI-Manager";
          onPrimary = () => { save(); window.PipelineSetup.installManager(); };
          secondary = "Skip for now";
        } else if (packs.length && d.needs_manager_restart) {
          primary = "Restart ComfyUI";
          onPrimary = () => { save(); window.PipelineSetup.restartComfy(); };
          secondary = "Skip for now";
        } else if (packs.length && d.manager_available) {
          primary = "Install missing nodes";
          onPrimary = () => { save(); window.PipelineSetup.installPacks(d); };
          secondary = "Skip for now";
        }
        box.append(actions({ primary, onPrimary, secondary, onSecondary }));
        return box;
      },
    },

    // ── text-to-video vs image-to-video ─────────────────────────────────────
    gentype: {
      render() {
        const box = el("div", "oo-step oo-step-wide");
        box.append(head("gentype", "How do you want to start a shot?",
          "This only sets what the editor asks you for. You can use both in the same project."));

        const cards = el("div", "oo-cards oo-cards-2");
        let bar;
        const paint = () => {
          [...cards.children].forEach((c) => c.classList.toggle("active", c.dataset.key === ctx.genType));
          const nb = mkBar(); bar.replaceWith(nb); bar = nb;
        };
        const card = (key, title, sub) => {
          const c = el("button", "oo-card");
          c.type = "button"; c.dataset.key = key;
          c.append(el("div", "oo-card-title", title));
          c.append(el("div", "oo-card-sub", sub));
          c.onclick = () => { ctx.genType = key; paint(); };
          return c;
        };
        cards.append(
          card("t2v", "From a prompt",
            "Text to video. No image is needed anywhere — anchors and guides stay out of your way."),
          card("i2v", "From an image",
            "Image to video. Each shot starts from a still, and scenes can carry the previous frame forward."),
        );
        box.append(cards);
        bar = mkBar();
        box.append(bar);
        function mkBar() {
          return actions({ disabled: !ctx.genType, onPrimary: saveGenType });
        }
        if (ctx.genType) paint();
        return box;
      },
    },

    // ── model files ─────────────────────────────────────────────────────────
    models: {
      async enter() {
        try { await S().loadModels?.(); } catch (_) {}
      },
      render() {
        const box = el("div", "oo-step");
        const own = ctx.family === "__own__";
        box.append(head("models", own ? "Wire your pipeline" : "Point it at your models",
          own
            ? "Nothing is generated until a final IMAGE reaches the global video output."
            : "Pick the checkpoint, text encoder and VAE files this model uses. Models opens over this screen."));

        const p = el("p", "oo-body");
        p.textContent = ctx.genType === "t2v"
          ? "Set for text-to-video, so no image loader is required."
          : "Set for image-to-video, so a still is expected at the start of a shot.";
        box.append(p);

        box.append(actions({
          primary: "Open Models…",
          onPrimary: () => window.ModelsModal.open(),
          secondary: "Continue",
          onSecondary: next,
        }));
        return box;
      },
    },

    // ── optional extras ─────────────────────────────────────────────────────
    extras: {
      render() {
        const wanted = host().extras;
        const box = el("div", "oo-step oo-step-wide");
        box.append(head("extras", "Optional, and easy to add later",
          "None of this is needed to generate. Skip it and come back when you want it."));

        const ROWS = {
          links: {
            title: "Pipeline links",
            sub: "Drive several node inputs from one control, or tie an input to a project value.",
            open: () => window.SettingsWindow.open("models"),
          },
          shortcuts: {
            title: "Prompt shortcuts",
            sub: "Short triggers that expand into full phrases while you type.",
            open: () => window.SettingsWindow.open("shortcuts"),
          },
          splits: {
            title: "Prompt splits",
            sub: "Markers that cut one written prompt into separate scenes.",
            open: () => window.Composer?.open?.(),
          },
        };
        const list = el("div", "oo-rows");
        wanted.forEach((k) => {
          const r = ROWS[k];
          if (!r) return;
          const row = el("div", "oo-row");
          const main = el("div", "oo-row-main");
          main.append(el("div", "oo-row-title", r.title));
          main.append(el("div", "oo-row-sub", r.sub));
          row.append(main);
          const b = el("button", "oo-btn oo-btn-small", "Set up");
          b.type = "button";
          b.onclick = () => { save(); close(); r.open(); };
          row.append(b);
          list.append(row);
        });
        box.append(list);
        box.append(actions({ primary: "Skip for now", onPrimary: next }));
        return box;
      },
    },

    // ── done ────────────────────────────────────────────────────────────────
    done: {
      render() {
        const box = el("div", "oo-step");
        box.append(head("done", "You're ready to generate",
          isEasyGen()
            ? "Write a prompt at the bottom and press Generate."
            : "Write in the Composer, then press Generate in the timeline header."));
        box.append(actions({ primary: "Let's go", onPrimary: finish }));
        return box;
      },
    },
  };

  // ── the splash, which is not a step: it has no Back and no progress ───────

  function splash() {
    const box = el("div", "oo-splash");
    const mark = el("div", "oo-splash-mark", "◉");
    box.append(mark);
    box.append(el("h1", "oo-splash-title", "Welcome to FunPack"));
    box.append(el("p", "oo-splash-lead", host().lead));

    const begin = el("button", "oo-btn oo-btn-primary oo-btn-hero", "Begin");
    begin.type = "button";
    begin.onclick = () => { idx = 0; renderStep(); };
    box.append(begin);

    const load = el("button", "oo-btn oo-btn-quiet", "Load an existing project");
    load.type = "button";
    load.onclick = openExisting;
    box.append(load);

    box.append(window.FunPackGit.maintenanceRow("oo-maint"));

    // Return advances, the way the Setup Assistant's splash does.
    const onKey = (e) => {
      if (e.key === "Enter" && root && root.querySelector(".oo-splash")) {
        e.preventDefault();
        begin.click();
      }
    };
    document.addEventListener("keydown", onKey);
    box._cleanup = () => document.removeEventListener("keydown", onKey);
    setTimeout(() => begin.focus(), 80);
    return box;
  }

  // ── actions ───────────────────────────────────────────────────────────────

  async function createProject() {
    const name = (ctx.name || "").trim() || (isEasyGen() ? "Untitled" : "Untitled montage");
    const bar = root.querySelector(".oo-actions");
    const again = !!ctx.project;   // stepped Back to this screen and pressed Continue again
    if (bar) clear(bar).append(busy(again ? "Saving…" : "Creating project…"));
    try {
      if (again) {
        // Don't leave a trail of abandoned projects behind a user who changes their mind.
        if (S().get().project?.name !== name) S().patchProject({ name });
      } else {
        await host().newProject(name);
      }
      ctx.project = S().get().project;
      if (ctx.family === "__own__") await window.PipelineSetup.useOwnPipeline();
      else await window.PipelineSetup.applyFamily(ctx.family);
    } catch (e) {
      if (bar) { clear(bar); bar.append(el("div", "oo-error", "Could not create the project: " + (e.message || e))); }
      return;
    }
    next();
  }

  function saveGenType() {
    try { S().setEditorSetting?.("generationType", ctx.genType); } catch (_) {}
    next();
  }

  function openExisting() {
    close();
    if (isEasyGen()) window.ProjectMenu.open({ dismissable: true });
    else window.WelcomePage.open();
  }

  // ── navigation ────────────────────────────────────────────────────────────

  async function go(delta) {
    let i = idx + delta;
    while (i >= 0 && i < steps.length) {
      const st = STEPS[steps[i]];
      if (st.enter) await st.enter();
      if (!st.skip || !st.skip()) break;
      i += delta;       // stepping over a skipped screen keeps Back symmetrical
    }
    if (i >= steps.length) { finish(); return; }
    idx = Math.max(0, i);
    renderStep();
  }

  const next = () => go(1);
  const back = () => go(-1);

  function renderStep() {
    const name = steps[idx];
    const st = STEPS[name];
    paint(st.render(), name);
    save();
  }

  function paint(node, name) {
    if (stage?._child?._cleanup) stage._child._cleanup();
    clear(stage);
    root.querySelector(".oo-actions")?.remove();
    stage.dataset.step = name || "splash";
    stage._child = node;
    stage.append(node);
    // The action bar is position:fixed, and the step's entry animation leaves a resolved
    // (identity) transform behind, which would make the step a containing block and trap
    // the bar inside it. Steps still declare their own bar; it just gets hoisted out.
    const bar = node.querySelector(":scope > .oo-actions");
    if (bar) root.append(bar);
    root.classList.toggle("oo-has-actions", !!bar);
    // Re-trigger the entry animation on every step, not just the first.
    stage.classList.remove("oo-in");
    void stage.offsetWidth;
    stage.classList.add("oo-in");
    const backBtn = root.querySelector(".oo-back");
    backBtn.hidden = !name || idx <= 0;
  }

  // ── persistence: an install restarts ComfyUI and reloads the page ─────────

  function save() {
    try {
      localStorage.setItem("funpack_wizard", JSON.stringify({
        idx, ctx: { name: ctx.name, family: ctx.family, genType: ctx.genType, projectId: ctx.project?.id || null },
      }));
    } catch (_) {}
  }

  function restore() {
    try {
      const raw = localStorage.getItem("funpack_wizard");
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (_) { return null; }
  }

  function clearSaved() {
    try { localStorage.removeItem("funpack_wizard"); } catch (_) {}
  }

  function finish() {
    try { localStorage.setItem(LS_DONE, "1"); } catch (_) {}
    clearSaved();
    // The wizard IS the pipeline setup — without this its modal reopens over the editor
    // and asks for the family and the node packs all over again.
    window.PipelineSetup?.markHandled?.();
    close();
  }

  // ── lifecycle ─────────────────────────────────────────────────────────────

  function close() {
    if (stage?._child?._cleanup) stage._child._cleanup();
    root?.remove();
    root = null;
    stage = null;
  }

  function open({ resume } = {}) {
    close();
    steps = host().steps;
    // project stays null until the wizard creates one: seeding it from whatever happens
    // to be open would make the project step rename that project instead of making a new one.
    ctx = { name: "", family: null, genType: null, project: null, deps: null };

    root = el("div", "oo-root");
    const backBtn = el("button", "oo-back", "‹");
    backBtn.type = "button";
    backBtn.title = "Back";
    backBtn.hidden = true;
    backBtn.onclick = back;
    root.append(backBtn);

    stage = el("div", "oo-stage");
    root.append(stage);
    document.body.append(root);

    const saved = resume ? restore() : null;
    if (saved) {
      Object.assign(ctx, saved.ctx || {});
      idx = Math.min(Math.max(saved.idx || 0, 0), steps.length - 1);
      // The project survived the restart; the in-memory handle did not.
      const pid = saved.ctx?.projectId;
      const load = pid ? S().loadProject(pid).catch(() => {}) : Promise.resolve();
      load.then(() => { ctx.project = S().get().project; renderStep(); });
      return;
    }
    idx = 0;
    paint(splash(), null);
  }

  function done() {
    try { return localStorage.getItem(LS_DONE) === "1"; } catch (_) { return false; }
  }

  // Called by each host's boot instead of its old welcome screen.
  function maybeOpen() {
    if (window.__FUNPACK_TOUR__) return false;
    if (restore()) { open({ resume: true }); return true; }
    if (done()) return false;
    open();
    return true;
  }

  window.Onboarding = { open, close, maybeOpen, done, isOpen: () => !!root };
})();
