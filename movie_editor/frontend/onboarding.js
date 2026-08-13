// Setup wizard and start screen.
//
// Shape is lifted from the macOS Setup Assistant: full screen, no window chrome, one
// vertically centred column per step, a glyph over a large headline over a short
// paragraph, controls in the middle, and a fixed action bar at the bottom whose primary
// button is always the way forward. Back is a chevron in the top-left; there are no step
// dots, because a linear flow that tells you how far you have to go invites you to bail.
//
(function () {
  const { el, clear } = window.dom;
  const S = () => window.Store;
  const API = () => window.MovieEditorAPI;
  const LS_DONE = "funpack_onboarded";
  const LS_RUN = "funpack_wizard";       // an unfinished run (see save())

  let root = null;       // the full-screen overlay
  let stage = null;      // the animated content column
  let idx = 0;
  let steps = [];
  let ctx = null;        // per-run state: project, family, deps, generation type
  let panel = null;      // { kind, cleanup } while a hosted pane is on screen

  // One app now — Simple and Editor are modes of it, and setup is the same either way.
  const HOST = {
    lead: "Multi-scene video on a real timeline.",
    steps: ["theme", "uimode", "project", "prereqs", "gentype", "models", "extras", "tour", "done"],
    extras: ["links", "shortcuts", "splits"],
    newProject: (name) => S().newProject(name),
  };
  const host = () => HOST;

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
    uimode: '<rect x="5" y="9" width="38" height="30" rx="4" fill="none" stroke="currentColor" stroke-width="2.4"/>'
      + '<path d="M24 9v30" stroke="currentColor" stroke-width="2.4"/>'
      + '<path d="M10 17h8M10 23h8" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>',
    tour: '<circle cx="24" cy="24" r="17" fill="none" stroke="currentColor" stroke-width="2.4"/>'
      + '<path d="M20 16.5l12 7.5-12 7.5z" fill="currentColor"/>',
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

  // ── hosted panes ──────────────────────────────────────────────────────────
  // Models, links, shortcuts and splits, shown as full screens of the wizard instead of
  // handing off to the Settings window or a floating Composer. Same builders and editors,
  // one Done button back. Panes are NOT steps: they open over the current step and return
  // to it, so Back stays symmetrical and the step count never shifts.
  const PANES = {
    models: {
      title: "Models & pipeline",
      sub: "Point each loader at a file you have. Nothing is saved until you press Save on a node.",
      mount: (body, setActions) => window.SettingsWindow.mountSection("models", body, { setActions }),
    },
    links: {
      title: "Pipeline links",
      sub: "Drive several node inputs from one control, or tie an input to a project value.",
      mount: (body, setActions) => window.SettingsWindow.mountSection("models", body, { setActions }),
    },
    shortcuts: {
      title: "Prompt shortcuts",
      sub: "Short triggers that expand into full phrases while you type.",
      mount: (body) => ({ cleanup: window.Composer.mountPane("shortcuts", body) }),
    },
    splits: {
      title: "Prompt splits",
      sub: "Markers that cut one written prompt into separate scenes.",
      mount: (body) => ({ cleanup: window.Composer.mountPane("splits", body) }),
    },
    projects: {
      title: "Your projects",
      sub: "Open one, or bring in a project file from somewhere else.",
      mount: (body, setActions) => mountProjects(body, setActions),
    },
  };

  // The project picker. This screen replaced a separate welcome card — two different
  // front doors for the same decision was one too many.
  function mountProjects(body, setActions) {
    const importInput = el("input");
    importInput.type = "file";
    importInput.accept = ".json,.funpack_project.json,application/json";
    importInput.style.display = "none";
    importInput.onchange = async () => {
      const file = importInput.files?.[0];
      importInput.value = "";
      if (!file) return;
      await S().importProject(file);
      close();
    };
    body.append(importInput);

    const imp = el("button", "btn ghost tiny", "Import project file…");
    imp.type = "button";
    imp.onclick = () => importInput.click();
    setActions([imp]);

    const list = el("div", "oo-projects");
    body.append(list);

    const paintList = async () => {
      clear(list);
      let projects = [];
      try { projects = (await API().listProjects()).projects || []; } catch (_) {}
      projects.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
      if (!projects.length) {
        list.append(el("p", "oo-body", "No projects yet. Go back and press Begin."));
        return;
      }
      projects.forEach((p) => {
        const row = el("button", "oo-project");
        row.type = "button";
        const main = el("div", "oo-project-main");
        main.append(el("div", "oo-project-name", p.name));
        const n = p.scene_count || 0;
        main.append(el("div", "oo-project-meta",
          `${n} ${n === 1 ? "scene" : "scenes"} · ${when(p.updated_at)}`));
        row.append(main);
        row.onclick = () => { close(); S().loadProject(p.id); };
        list.append(row);
      });
    };
    paintList();
    return { cleanup: () => {} };
  }

  function when(ts) {
    if (!ts) return "never opened";
    const days = Math.floor((Date.now() / 1000 - ts) / 86400);
    if (days <= 0) return "today";
    if (days === 1) return "yesterday";
    if (days < 30) return days + " days ago";
    return new Date(ts * 1000).toLocaleDateString();
  }

  function openPane(kind) {
    const spec = PANES[kind];
    if (!spec) return;
    const fromSplash = !panel && !!root?.querySelector(".oo-splash");
    closePane();
    const box = el("div", "oo-pane");
    const head = el("div", "oo-pane-head");
    const ht = el("div", "oo-pane-head-text");
    ht.append(el("h2", "oo-pane-title", spec.title));
    ht.append(el("p", "oo-pane-sub", spec.sub));
    const tools = el("div", "oo-pane-tools");
    head.append(ht, tools);
    box.append(head);
    const body = el("div", "oo-pane-body");
    box.append(body);

    const mounted = spec.mount(body, (nodes) => {
      clear(tools);
      (nodes || []).forEach((n) => tools.append(n));
    });
    if (!mounted) {
      body.append(el("p", "oo-body", "This part of the setup isn't available in this build."));
    }

    panel = { kind, fromSplash, cleanup: mounted?.cleanup || null };
    paint(box, "pane:" + kind, { pane: true });
    root.append(actions({ primary: fromSplash ? "Back" : "Done", onPrimary: closePane }));
    root.classList.add("oo-has-actions");
  }

  function closePane() {
    if (!panel) return;
    const backToSplash = panel.fromSplash;
    try { panel.cleanup?.(); } catch (_) {}
    panel = null;
    if (!root) return;
    if (backToSplash) paint(splash(), null);
    else renderStep();
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

    // ── how much of the app to show ─────────────────────────────────────────
    uimode: {
      render() {
        const box = el("div", "oo-step oo-step-wide");
        box.append(head("uimode", "How much do you want in front of you?",
          "You can switch at any time from the control beside the FunPack wordmark."));

        const cards = el("div", "oo-cards oo-cards-2");
        let bar;
        const paint = () => {
          [...cards.children].forEach((c) => c.classList.toggle("active", c.dataset.key === ctx.uiMode));
          const nb = mkBar(); bar.replaceWith(nb); bar = nb;
        };
        const card = (key, title, sub) => {
          const c = el("button", "oo-card");
          c.type = "button"; c.dataset.key = key;
          c.append(el("div", "oo-card-title", title));
          c.append(el("div", "oo-card-sub", sub));
          c.onclick = () => { ctx.uiMode = key; paint(); };
          return c;
        };
        cards.append(
          card("simple", "Simple",
            "One prompt and Generate. Rating-driven steering, cross-shot memory and "
            + "experimental sampling stay off, so runs are quicker and there is less to read."),
          card("editor", "Editor",
            "The full cutting room: timeline, ratings, and every Studio and sampler setting."),
        );
        box.append(cards);
        bar = mkBar();
        box.append(bar);
        function mkBar() {
          return actions({ disabled: !ctx.uiMode, onPrimary: saveUiMode });
        }
        if (ctx.uiMode) paint();
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
        input.value = ctx.name || "Untitled montage";
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
          onPrimary = () => { saveForRestart(); window.PipelineSetup.installManager(); };
          secondary = "Skip for now";
        } else if (packs.length && d.needs_manager_restart) {
          primary = "Restart ComfyUI";
          onPrimary = () => { saveForRestart(); window.PipelineSetup.restartComfy(); };
          secondary = "Skip for now";
        } else if (packs.length && d.manager_available) {
          primary = "Install missing nodes";
          onPrimary = () => { saveForRestart(); window.PipelineSetup.installPacks(d); };
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
            : "Pick the checkpoint, text encoder and VAE files this model uses."));

        const p = el("p", "oo-body");
        p.textContent = ctx.genType === "t2v"
          ? "Set for text-to-video, so no image loader is required."
          : "Set for image-to-video, so a still is expected at the start of a shot.";
        box.append(p);

        box.append(actions({
          primary: "Set up models…",
          onPrimary: () => openPane("models"),
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

        const list = el("div", "oo-rows");
        wanted.forEach((k) => {
          const r = PANES[k];
          if (!r) return;
          const row = el("div", "oo-row");
          const main = el("div", "oo-row-main");
          main.append(el("div", "oo-row-title", r.title));
          main.append(el("div", "oo-row-sub", r.sub));
          row.append(main);
          const b = el("button", "oo-btn oo-btn-small", "Set up");
          b.type = "button";
          // Opens as a screen of the wizard, not as a window over it — Done comes straight
          // back here with Skip/Continue still where you left them.
          b.onclick = () => openPane(k);
          row.append(b);
          list.append(row);
        });
        box.append(list);
        box.append(actions({ primary: "Skip for now", onPrimary: next }));
        return box;
      },
    },

    // ── guided tour ─────────────────────────────────────────────────────────
    tour: {
      // The tour runs in its own page mode against a sandbox project, so it cannot start
      // over the wizard: the choice is recorded here and acted on by finish().
      render() {
        const box = el("div", "oo-step");
        box.append(head("tour", "Want the guided tour?",
          "A walk through the timeline, the Composer and Generate, on a sandbox project. "
          + "Nothing you do in it touches your own work."));
        const p = el("p", "oo-body",
          "Choosing to watch it starts the tour as soon as you finish setup. You can run it "
          + "again at any time from Help ▸ Welcome tour.");
        box.append(p);
        box.append(actions({
          primary: "Watch tour",
          onPrimary: () => { ctx.wantTour = true; next(); },
          secondary: "Skip the tour",
          onSecondary: () => { ctx.wantTour = false; next(); },
        }));
        return box;
      },
    },

    // ── done ────────────────────────────────────────────────────────────────
    done: {
      render() {
        const box = el("div", "oo-step");
        box.append(head("done", "You're ready to generate",
          ctx.wantTour
            ? "The guided tour starts as soon as you press the button."
            : window.FunPackMode?.isSimple()
              ? "Write a prompt, then press Generate."
              : "Write in the Composer, then press Generate in the timeline header."));
        box.append(actions({ primary: ctx.wantTour ? "Start the tour" : "Let's go", onPrimary: finish }));
        return box;
      },
    },
  };

  // ── the splash, which is not a step: it has no Back and no progress ───────

  // Most recently touched project, or null when there is none to go back to.
  async function lastProject() {
    try {
      const { projects } = await API().listProjects();
      return (projects || []).slice()
        .sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0))[0] || null;
    } catch (_) { return null; }
  }

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

    // One click back into whatever was open before the restart. Filled in when the
    // project list arrives — the splash paints first and must not wait on a fetch.
    const resumeBtn = el("button", "oo-btn oo-btn-quiet oo-btn-resume");
    resumeBtn.type = "button";
    resumeBtn.hidden = true;
    box.append(resumeBtn);

    const load = el("button", "oo-btn oo-btn-quiet", "Load an existing project");
    load.type = "button";
    load.onclick = openExisting;
    box.append(load);

    lastProject().then((p) => {
      if (!p || !resumeBtn.isConnected) return;
      resumeBtn.textContent = "Continue with “" + p.name + "”";
      resumeBtn.hidden = false;
      resumeBtn.onclick = () => {
        close();
        (S().loadProject(p.id) || Promise.resolve()).catch(() => {});
      };
    });

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
    const name = (ctx.name || "").trim() || "Untitled montage";
    const bar = root.querySelector(".oo-actions");
    // An abandoned run left a project behind: adopt it rather than making a second one.
    // Failing to load it (deleted since) just falls through to creating a fresh project.
    if (!ctx.project && ctx.carryProjectId) {
      if (bar) clear(bar).append(busy("Picking up where you left off…"));
      try {
        await S().loadProject(ctx.carryProjectId);
        if (S().get().project?.id === ctx.carryProjectId) ctx.project = S().get().project;
      } catch (_) {}
      ctx.carryProjectId = null;
    }
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

  function saveUiMode() {
    try { window.FunPackMode?.set(ctx.uiMode); window.FunPackMode?.markWarned(); } catch (_) {}
    next();
  }

  function saveGenType() {
    try { S().patchProject?.({ generation_mode: ctx.genType }); } catch (_) {}
    next();
  }

  function openExisting() { openPane("projects"); }

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
    save(false);
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
    // On a hosted pane the chevron closes the pane rather than stepping back — the pane
    // is not a step, and leaving it is the only "back" that means anything there.
    backBtn.hidden = !name || (!panel && idx <= 0);
    root.classList.toggle("oo-pane-mode", !!panel);
  }

  // ── persistence ───────────────────────────────────────────────────────────
  // `resumable` marks the one case that resumes mid-flow: the wizard restarted ComfyUI
  // itself and the page is about to reload. A refresh or a closed tab starts over, and
  // carries the project the run had already created. Cleared only by finish().
  function save(resumable) {
    try {
      localStorage.setItem(LS_RUN, JSON.stringify({
        idx: resumable ? idx : 0,
        resumable: !!resumable,
        ctx: { name: ctx.name, family: ctx.family, genType: ctx.genType, uiMode: ctx.uiMode,
               wantTour: ctx.wantTour, projectId: ctx.project?.id || null },
      }));
    } catch (_) {}
  }

  // Called at the three points that hand off to a ComfyUI restart, and nowhere else.
  const saveForRestart = () => save(true);

  function restore() {
    try {
      const raw = localStorage.getItem(LS_RUN);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (_) { return null; }
  }

  function clearSaved() {
    try { localStorage.removeItem(LS_RUN); } catch (_) {}
  }

  // The tour lives in its own page mode against a sandbox project, so it cannot start
  // over the editor — finishing navigates there.
  function startTour() {
    try {
      const u = new URL(window.location.href);
      u.searchParams.set("mode", "tour");
      window.location.href = u.pathname + u.search;
      return true;
    } catch (_) { return false; }
  }

  function finish() {
    try { localStorage.setItem(LS_DONE, "1"); } catch (_) {}
    clearSaved();
    // The wizard IS the pipeline setup — without this its modal reopens over the editor
    // and asks for the family and the node packs all over again.
    window.PipelineSetup?.markHandled?.();
    const tour = !!ctx.wantTour;
    close();
    if (tour) startTour();
  }

  // ── lifecycle ─────────────────────────────────────────────────────────────

  function close() {
    if (panel) { try { panel.cleanup?.(); } catch (_) {} panel = null; }
    if (stage?._child?._cleanup) stage._child._cleanup();
    root?.remove();
    root = null;
    stage = null;
  }

  // `resume` picks up mid-flow after the wizard's own ComfyUI restart. `carry` is an
  // abandoned run's leftovers — its project comes back so a second attempt renames that
  // one instead of leaving it behind, but the flow itself starts at the splash.
  function open({ resume, carry } = {}) {
    close();
    steps = host().steps;
    // project stays null until the wizard creates one: seeding it from whatever happens
    // to be open would make the project step rename that project instead of making a new one.
    ctx = { name: "", family: null, genType: null, uiMode: null, wantTour: false, project: null, deps: null };

    root = el("div", "oo-root");
    const backBtn = el("button", "oo-back", "‹");
    backBtn.type = "button";
    backBtn.title = "Back";
    backBtn.hidden = true;
    backBtn.onclick = () => (panel ? closePane() : back());
    root.append(backBtn);

    stage = el("div", "oo-stage");
    root.append(stage);
    document.body.append(root);

    const saved = (resume || carry) ? restore() : null;
    if (saved && resume && saved.resumable) {
      Object.assign(ctx, saved.ctx || {});
      idx = Math.min(Math.max(saved.idx || 0, 0), steps.length - 1);
      // The project survived the restart; the in-memory handle did not.
      const pid = saved.ctx?.projectId;
      const load = pid ? S().loadProject(pid).catch(() => {}) : Promise.resolve();
      load.then(() => { ctx.project = S().get().project; renderStep(); });
      return;
    }
    idx = 0;
    if (saved && carry) {
      // Name and family are offered again as defaults; nothing is applied until the
      // project step is completed a second time.
      ctx.name = saved.ctx?.name || "";
      ctx.family = saved.ctx?.family || null;
      ctx.carryProjectId = saved.ctx?.projectId || null;
    }
    paint(splash(), null);
  }

  function done() {
    try { return localStorage.getItem(LS_DONE) === "1"; } catch (_) { return false; }
  }

  // The start screen for both frontends: shown whenever nothing is open, not just on a
  // first run. Its splash carries Begin, the last project, the project list, and the
  // maintenance actions, so there is one front door instead of two.
  function maybeOpen() {
    if (window.__FUNPACK_TOUR__) return false;
    if (S()?.get?.().project) return false;   // work already open — don't cover it
    const saved = restore();
    if (saved?.resumable) { open({ resume: true }); return true; }
    // An unfinished run that did NOT end in a deliberate restart: start over.
    open(saved ? { carry: true } : undefined);
    return true;
  }

  // Menu entry: run setup again on demand, whatever state the app is in.
  function reopen() {
    clearSaved();
    open();
  }

  window.Onboarding = { open, reopen, close, maybeOpen, done, isOpen: () => !!root };
})();
