// Welcome tour: sandbox project + spotlight guide + FAQ (?mode=tour).
(function () {
  if (!window.__FUNPACK_TOUR__) return;

  const { el } = window.dom;
  const TOUR_KEY = "funpack_tour_completed_v1";

  const FAQ = [
    {
      cat: "Setup",
      items: [
        {
          q: "I cannot finish model setup - dropdowns are empty.",
          a: "Use Settings - Refresh model list (or press R in ComfyUI) so loader combos pick up installed checkpoints. Confirm the files exist in your ComfyUI model folders, then restart ComfyUI if you just installed something new.",
        },
        {
          q: "Generate is disabled or fails right away.",
          a: "Open FunPack in the menu bar and check Workflow template: it must say loaded. Export your graph from ComfyUI with Save (API Format), not the regular UI workflow JSON. Also fill every required slot under Settings - Models.",
        },
        {
          q: "Missing media for continuity or guide stack.",
          a: "A bin asset referenced by a guide was deleted or moved. Re-upload the file in Media, or clear the broken guide in the Inspector.",
        },
      ],
    },
    {
      cat: "Generation and preview",
      items: [
        {
          q: "My scene says Not generated yet in preview.",
          a: "Run Generate or Generate Selected after setup. Pending generative scenes are skipped in Render until they have a render, unless you replace them with a video clip from the media bin.",
        },
        {
          q: "The scene looks stale after I changed the prompt.",
          a: "Re-generate that scene. A stale badge means preview still shows an older render from before your edit.",
        },
        {
          q: "Continuity broke between scenes.",
          a: "Check transition markers in your montage text, the project anchor, and Engine settings for auto continuity. Try Mixed or Carry source on later scenes.",
        },
      ],
    },
    {
      cat: "Render vs Export",
      items: [
        {
          q: "I clicked Export but only one scene was saved.",
          a: "Export (timeline toolbar) saves only the selected clip(s), hard-cut, with no full audio mix. For the whole movie, use Render in the timeline header, not Export.",
        },
        {
          q: "Render says nothing to render.",
          a: "You need at least one generated clip, or overlay or audio content (blank-canvas render). Generate first, or add overlays or audio tracks.",
        },
        {
          q: "Where did my final file go?",
          a: "Render downloads through the browser. Also check ComfyUI output folders if the server logged a path.",
        },
      ],
    },
    {
      cat: "Timeline editing",
      items: [
        {
          q: "Split (S) does nothing.",
          a: "Select a clip or audio block first. The playhead must be inside it, or the split uses the clip midpoint.",
        },
        {
          q: "Audio disappeared when I removed the video.",
          a: "Use Separate audio before deleting if you want to keep the sound. Separated audio stays on its own lane.",
        },
        {
          q: "Overlays do not show in preview.",
          a: "Overlays appear when the playhead is inside their time range. Check duration and lane visibility.",
        },
      ],
    },
    {
      cat: "Ratings and Studio",
      items: [
        {
          q: "What do scene ratings do?",
          a: "After a render, rate a generative scene. FunPack Studio uses ratings on the next Generate to steer quality (requires the built-in Studio pipeline).",
        },
      ],
    },
  ];

  const CHAPTERS = [
    { id: "start", label: "Start", step: 0 },
    { id: "setup", label: "Setup", step: 3 },
    { id: "timeline", label: "Timeline", step: 7 },
    { id: "actions", label: "Generate and Render", step: 12 },
    { id: "libraries", label: "Libraries", step: 15 },
    { id: "faq", label: "FAQ", step: 17 },
  ];

  function demoProject() {
    const now = Date.now() / 1000;
    return {
      id: "tour-demo-project",
      name: "Demo Montage",
      anchor: "A lone traveler in warm morning light, cinematic realism.",
      global_prompt: "",
      intro_transition: "scene 1",
      negative_prompt: "",
      seed: 42,
      num_frames_per_scene: 97,
      frame_rate: 25,
      width: 768,
      height: 512,
      max_scenes: 8,
      conditioning_slot: "funpack",
      sampler_slot: "funpack",
      studio_inputs: {},
      sampler_inputs: {},
      keep_original_audio: true,
      scenes: [
        {
          id: "tour-s1",
          text: "Wide establishing shot of the city at dawn, soft haze.",
          transition_to_next: "cut",
          transition_frames: 0,
          video_transition: "",
          effects: {},
          audio_volume: 1,
          audio_separated: false,
          frames_mode: "project",
          fps_mode: "project",
          source: { type: "empty", media_ref: null },
          rating: "good",
          excluded: false,
          gen_unit_id: "tour-s1",
          cut_offset_frames: 0,
          guides: [],
          source_in: 0,
          source_dur: null,
        },
        {
          id: "tour-s2",
          text: "",
          transition_to_next: "dissolve",
          transition_frames: 12,
          video_transition: "crossfade",
          effects: {},
          audio_volume: 1,
          audio_separated: true,
          frames_mode: "project",
          fps_mode: "project",
          source: { type: "video", media_ref: "tour-media-video" },
          rating: "",
          excluded: false,
          gen_unit_id: "tour-s2",
          cut_offset_frames: 0,
          guides: [],
          source_in: 0.5,
          source_dur: 4.2,
        },
        {
          id: "tour-s3",
          text: "Close-up reaction, emotional beat, shallow depth of field.",
          transition_to_next: "",
          transition_frames: 0,
          video_transition: "",
          effects: {},
          audio_volume: 1,
          audio_separated: false,
          frames_mode: "project",
          fps_mode: "project",
          source: { type: "carry", media_ref: null },
          rating: "",
          excluded: false,
          gen_unit_id: "tour-s3",
          cut_offset_frames: 0,
          guides: [],
          source_in: 0,
          source_dur: null,
        },
      ],
      audio_tracks: [
        {
          id: "tour-audio-sep",
          kind: "separated",
          scene_id: "tour-s2",
          start_sec: 3.88,
          source_in_sec: 0,
          source_dur: 4.2,
          volume: 1,
          label: "S2 separated",
        },
        {
          id: "tour-audio-music",
          kind: "overlay",
          media_ref: "tour-media-audio",
          start_sec: 0,
          source_in_sec: 0,
          source_dur: 12,
          volume: 0.85,
          label: "Music bed",
        },
      ],
      overlay_lanes: [{ id: "tour-ovl-lane-1", label: "Titles" }],
      overlay_tracks: [
        {
          id: "tour-ovl-text",
          lane_id: "tour-ovl-lane-1",
          kind: "text",
          text: "Demo title",
          start_sec: 0.2,
          dur_sec: 2.5,
          x: 0.5,
          y: 0.12,
          width_px: 420,
          height_px: 80,
          keep_aspect: false,
          opacity: 1,
          font_size: 42,
          color: "#ffffff",
          align: "center",
        },
      ],
      models: {
        slots: [
          { id: "tour-slot-unet", type: "unet", node_class: "UNETLoader", label: "Main diffusion model", inputs: { unet_name: "demo_model.safetensors" } },
          { id: "tour-slot-vae", type: "vae", node_class: "VAELoader", label: "Video VAE", inputs: { vae_name: "demo_vae.safetensors" } },
          { id: "tour-slot-clip", type: "clip", node_class: "CLIPLoader", label: "Text encoder", inputs: { clip_name: "demo_clip.safetensors" } },
        ],
      },
      guide_settings: { stack_enabled: false },
      continuity_settings: {},
      generation_meta: {},
      scene_renders: {
        "tour-s1": {
          media: { filename: "tour_demo_render.mp4", subfolder: "funpack_tour", type: "output" },
          inSec: 0,
        },
      },
      scene_ghosts: [],
      created_at: now,
      updated_at: now,
    };
  }

  function demoState() {
    const project = demoProject();
    return {
      projects: [{ id: project.id, name: project.name, scene_count: project.scenes.length, updated_at: project.updated_at }],
      project,
      selectedSceneId: "tour-s1",
      selectedSceneIds: ["tour-s1"],
      selectedOverlayId: null,
      selectedAudioTrackId: null,
      transitions: [
        { name: "cut", label: "Cut" },
        { name: "dissolve", label: "Dissolve" },
      ],
      nleEffects: [{ id: "blur", label: "Blur" }],
      nleVideoTransitions: [{ id: "crossfade", label: "Crossfade" }],
      health: {
        ok: true,
        reference_loaded: true,
        template_exists: true,
        configured_slots: 3,
        comfy_url: window.location.origin,
      },
      preview: {
        combined_prompt: "Demo combined prompt preview",
        parsed: { scenes: 3 },
        warning: "",
        parse_error: "",
      },
      gen: { state: "idle", promptId: null, media: [], msg: "" },
      sceneRenders: JSON.parse(JSON.stringify(project.scene_renders)),
      sceneGhosts: [],
      notice: "",
      saving: false,
      unsaved: false,
      models: JSON.parse(JSON.stringify(project.models)),
      mediaBin: [
        { id: "tour-media-video", name: "B-roll alley.mp4", kind: "video", duration_sec: 8.4 },
        { id: "tour-media-image", name: "Reference still.jpg", kind: "image" },
        { id: "tour-media-audio", name: "Ambient bed.wav", kind: "audio", duration_sec: 30 },
      ],
      mediaPreviewId: null,
      shortcuts: [{ id: "tour-sh-1", name: "golden hour", triggers: ["@golden"], replacements: ["warm sunset light"] }],
      imageTargets: [],
      ratingLabels: ["good", "great", "bad"],
    };
  }

  async function bootstrap(state, notify) {
    const demo = demoState();
    Object.assign(state, demo);
    try { window.Store?.ensureOverlayLanes?.(); } catch (_) {}
    if (window.EditorHistory) window.EditorHistory.clear();
    try { window.Store?.syncGlobalPromptFromTimeline?.(); } catch (_) {}
    try { window.Store?.refreshPreview?.(); } catch (_) {}
    notify();
  }

  function patchApi() {
    const API = window.MovieEditorAPI;
    if (!API) return;
    const demo = demoState();
    const ok = (data) => () => Promise.resolve(JSON.parse(JSON.stringify(data)));
    API.health = ok(demo.health);
    API.preview = ok(demo.preview);
    API.getModels = ok(demo.models);
    API.saveModels = ok({});
    API.pipelinePorts = ok({ ports: [], core_producers: [], requirements: [], wiring: {} });
    API.coreGraph = ok({ nodes: [] });
    API.nodeRoles = ok({ roles: [] });
    API.allNodes = ok({ nodes: [] });
    API.nodeSpec = ok({ class_type: "DemoNode", inputs: {} });
    API.nodeCandidates = ok({ candidates: [] });
    API.ratingLabels = ok({ labels: demo.ratingLabels });
    API.listProjects = ok({ projects: demo.projects });
    API.listMedia = ok({ media: demo.mediaBin });
    API.transitions = ok({ transitions: demo.transitions });
    API.shortcuts = ok({ shortcuts: demo.shortcuts });
    API.nleLibrary = ok({ effects: demo.nleEffects, video_transitions: demo.nleVideoTransitions });
    API.refreshModels = ok({});
    API.generate = ok({ prompt_id: "tour-demo" });
    API.renderFinal = ok({ job_id: "tour-demo" });
    API.exportClip = ok({});
    API.exportClipsCombined = ok({ job_id: "tour-demo" });
    API.interrupt = ok({});
  }

  function patchStore(Store) {
    const blocked = [
      "generate", "generateMontage", "generateSelected", "renderFinal", "exportSelected",
      "saveSelectedToMediaBin", "commit", "newProject", "loadProject", "deleteProject",
      "importProject", "downloadProject", "uploadMedia", "deleteMedia", "deleteMediaMany",
      "interrupt", "resetStudioSession",
    ];
    const labels = {
      generate: "Generate",
      generateMontage: "Generate",
      generateSelected: "Generate Selected",
      renderFinal: "Render",
      exportSelected: "Export",
      saveSelectedToMediaBin: "Save to media bin",
      commit: "Save project",
      newProject: "New project",
      loadProject: "Open project",
      deleteProject: "Delete project",
      importProject: "Import project",
      downloadProject: "Download project",
      uploadMedia: "Upload media",
      resetStudioSession: "Reset Studio session",
    };
    blocked.forEach((name) => {
      if (typeof Store[name] !== "function") return;
      Store[name] = function () {
        TourGuide.toast(`Demo only: ${labels[name] || name} is disabled in the tour.`);
        return Promise.resolve();
      };
    });
    if (Store.refreshProjectList) Store.refreshProjectList = async () => {};
  }

  const STEPS = [
    {
      id: "welcome",
      title: "Welcome to Cutting Room",
      body: "FunPack's timeline editor inside ComfyUI. You write scenes, generate video, trim on a timeline, then Render a final file. This walkthrough uses a demo project - nothing here touches your real files or queues jobs.",
      placement: "center",
    },
    {
      id: "layout",
      title: "Four main zones",
      target: "#workspace",
      body: "Media (left) holds projects and libraries. Preview (top right) is the program monitor. Inspector (bottom right) edits the selected clip or project settings. Timeline (bottom) is scenes, audio, and overlays.",
      pad: 8,
    },
    {
      id: "dock-tabs",
      title: "Panel visibility",
      target: "#dock-tabs",
      body: "Toggle Media, Preview, and Settings when you need more timeline space. Most editing is Timeline plus Inspector.",
      pad: 6,
    },
    {
      id: "models-menu",
      title: "Set up Models first",
      target: '.menu-btn[data-menu="Settings"]',
      body: "Before generating, open Settings - Models. Tell Cutting Room which ComfyUI loader nodes and checkpoint files to use. Without this, Generate does not know your hardware setup.",
      pad: 4,
      before: () => closeMenus(),
    },
    {
      id: "models-modal",
      title: "Models dialog",
      target: ".modal-overlay .modal",
      body: "Add each loader type (UNet, VAE, CLIP, and so on), pick the node class, then set widget values from dropdowns. Use Wire to when an output must feed another node or a pipeline port.",
      pad: 12,
      before: () => {
        closeMenus();
        closeModalOverlay();
        const p = window.ModelsModal?.open?.();
        if (p && typeof p.then === "function") {
          p.then(() => requestAnimationFrame(() => TourGuide.reposition()));
        }
      },
      after: () => closeModalOverlay(),
    },
    {
      id: "workflow-template",
      title: "ComfyUI workflow template (API format)",
      target: "#health-chip",
      body: "Cutting Room queues your graph from a ComfyUI export. In ComfyUI enable dev mode, then Save (API Format) - not the regular UI workflow JSON. Point the app at that file (or import it under Settings - Import ComfyUI Workflow). FunPack menu shows Workflow template: loaded when ready.",
      pad: 6,
      before: () => { closeMenus(); closeModalOverlay(); },
    },
    {
      id: "engine-settings",
      title: "Engine settings",
      target: '.menu-btn[data-menu="Settings"]',
      body: "Open Settings - Engine settings for continuity, guide stacks, and sampler overrides. Defaults are fine to start; revisit when scene-to-scene identity drifts.",
      pad: 4,
      before: () => closeMenus(),
    },
    {
      id: "scene-clips",
      title: "Scene clips",
      target: "#timeline-body .tl-track2",
      body: "Each block is a scene. Click to select. Drag edges to trim. S splits at the playhead. Cmd/Ctrl-click toggles multi-select.",
      pad: 10,
      before: (S) => {
        closeMenus();
        S.selectScene("tour-s1");
      },
    },
    {
      id: "inspector-scene",
      title: "Scene inspector",
      target: "#inspector-body",
      body: "Edit the prompt and source type (Empty, Image, Carry) for the selected scene. Deselect all clips to edit project anchor, seed, and global timing. The whole-montage global prompt lives in the Composer.",
      pad: 8,
    },
    {
      id: "transitions",
      title: "Transitions and seams",
      target: "#timeline-body .seam-cut",
      body: "Seams between clips are transition markers in your montage text (generation) plus optional video crossfades (post-render). Custom split markers live in the Composer.",
      pad: 8,
      optionalTarget: "#timeline-body",
    },
    {
      id: "video-clip",
      title: "Video clips",
      target: '#timeline-body [data-scene-id="tour-s2"], #timeline-body .clip.clip-video',
      body: "Drop video from Media onto the timeline for locked B-roll - no generation needed. Trim, separate audio, or convert back to a generative scene later.",
      pad: 6,
      before: (S) => S.selectScene("tour-s2"),
      optionalTarget: "#timeline-body",
    },
    {
      id: "overlays-audio",
      title: "Overlays and audio lanes",
      target: "#timeline-body .tl-overlay-lanes, #timeline-body .tl-audio-lanes",
      body: "Overlay tracks composite text or images on export. Audio lanes hold original scene audio, separated clip audio, or imported music. Trim audio blocks like video clips.",
      pad: 8,
      optionalTarget: "#timeline-body",
    },
    {
      id: "generate",
      title: "Generate",
      target: '[data-tour="generate-all"]',
      body: "Generate sends scenes to ComfyUI and fills renders. Selected runs only highlighted scenes. Requires Models plus a workflow template. This demo will not queue real jobs.",
      pad: 6,
      tryClick: true,
    },
    {
      id: "preview",
      title: "Preview monitor",
      target: "#preview-body",
      body: "Scrub the timeline or press Space to preview. Green minibar chips mean rendered. Overlays and audio play even when there is no video underneath.",
      pad: 10,
    },
    {
      id: "render-vs-export",
      title: "Render vs Export",
      target: '[data-tour="render-final"]',
      body: "Render (header) stitches the whole timeline into one final MP4 with audio mix and overlays. Export (toolbar) saves only selected clip(s), hard-cut - not the full movie. For a complete output, use Render.",
      pad: 6,
      alsoHighlight: '[data-tour="export-scene"]',
    },
    {
      id: "composer",
      title: "Composer",
      target: "#composer-btn",
      body: "Shortcuts and Split markers live in the Composer — a window you open from here. Create them once and reuse them across scenes instead of retyping. (Per-clip effects and video transitions are edited in Properties.)",
      pad: 6,
    },
    {
      id: "project-io",
      title: "Project files",
      target: '.menu-btn[data-menu="File"]',
      body: "Projects auto-save locally. File - Save Project File / Load Project File moves .json between machines. Open recent from the welcome screen when you return.",
      pad: 4,
      before: () => closeMenus(),
    },
    {
      id: "faq",
      title: "FAQ",
      type: "faq",
    },
  ];

  function closeMenus() {
    try { document.getElementById("menu-veil").hidden = true; } catch (_) {}
    document.querySelectorAll(".menu.open").forEach((m) => m.classList.remove("open"));
  }

  function closeModalOverlay() {
    document.querySelectorAll(".modal-overlay").forEach((n) => {
      if (n.closest?.("#tour-root")) return;
      n.remove();
    });
  }

  function tourUrl() {
    const u = new URL(window.location.href);
    u.searchParams.set("mode", "tour");
    return u.pathname + u.search;
  }

  function editorUrl() {
    const u = new URL(window.location.href);
    u.searchParams.delete("mode");
    const q = u.searchParams.toString();
    return u.pathname + (q ? "?" + q : "");
  }

  const TourGuide = {
    step: 0,
    root: null,
    spot: null,
    card: null,
    toastEl: null,
    resizeObs: null,
    tryHandler: null,

    toast(msg) {
      if (!this.toastEl) {
        this.toastEl = el("div", "tour-toast");
        document.body.appendChild(this.toastEl);
      }
      this.toastEl.textContent = msg;
      this.toastEl.classList.add("show");
      clearTimeout(this._toastT);
      this._toastT = setTimeout(() => this.toastEl?.classList.remove("show"), 3200);
    },

    start(at) {
      if (this.root) return;
      document.body.classList.add("tour-mode");
      this.step = typeof at === "number" ? at : 0;
      this.root = el("div", "tour-root");
      this.root.id = "tour-root";
      this.root.innerHTML = "";
      const backdrop = el("div", "tour-backdrop");
      this.spot = el("div", "tour-spotlight");
      this.card = el("div", "tour-card");
      backdrop.append(this.spot);
      this.root.append(backdrop);
      this.root.append(this.card);
      const banner = el("div", "tour-banner");
      banner.append(el("span", "tour-banner-label", "Welcome tour"));
      banner.append(el("span", "tour-banner-note", "Demo project - no real saves or generation"));
      const exitBtn = el("button", "btn ghost tiny tour-exit", "Exit tour");
      exitBtn.onclick = () => this.exit(false);
      banner.append(exitBtn);
      this.root.append(banner);
      document.body.appendChild(this.root);
      window.addEventListener("resize", this._onResize = () => {
        const step = STEPS[this.step];
        if (step?.type === "faq") {
          this.positionCardCenter();
          this.card.style.maxWidth = "640px";
          return;
        }
        this.renderStep();
      });
      window.addEventListener("scroll", this._onScroll = () => this.reposition(), true);
      this.renderStep();
    },

    exit(completed) {
      if (completed) {
        try { localStorage.setItem(TOUR_KEY, String(Date.now())); } catch (_) {}
      }
      this.teardown();
      window.location.href = editorUrl();
    },

    teardown() {
      closeModalOverlay();
      closeMenus();
      if (this.tryHandler) {
        document.removeEventListener("click", this.tryHandler, true);
        this.tryHandler = null;
      }
      window.removeEventListener("resize", this._onResize);
      window.removeEventListener("scroll", this._onScroll, true);
      this.root?.remove();
      this.root = null;
      this.toastEl?.remove();
      this.toastEl = null;
      document.body.classList.remove("tour-mode");
    },

    go(delta) {
      const next = this.step + delta;
      if (next < 0 || next >= STEPS.length) return;
      const cur = STEPS[this.step];
      if (cur?.after) cur.after(window.Store);
      this.step = next;
      this.renderStep();
    },

    jump(toStep) {
      if (toStep < 0 || toStep >= STEPS.length) return;
      STEPS[this.step]?.after?.(window.Store);
      this.step = toStep;
      this.renderStep();
    },

    reposition() {
      const step = STEPS[this.step];
      if (!step || step.type === "faq" || step.placement === "center") return;
      const rect = this.resolveTarget(step);
      if (rect) {
        this.spot.hidden = false;
        this.positionSpotlight(rect, step.pad || 8);
        this.positionCardNear(rect, step.placement || "below");
      }
    },

    renderStep() {
      const step = STEPS[this.step];
      if (!step) return;
      this.root?.querySelector(".tour-backdrop")?.classList.remove("faq-mode");
      if (this.tryHandler) {
        document.removeEventListener("click", this.tryHandler, true);
        this.tryHandler = null;
      }
      step.before?.(window.Store);

      if (step.type === "faq") {
        this.renderFaq(step);
        return;
      }

      clear(this.card);
      this.card.className = "tour-card";
      this.spot.hidden = step.placement === "center" || !step.target;
      this.root.querySelector(".tour-backdrop").hidden = false;

      const head = el("div", "tour-card-head");
      head.append(el("span", "tour-step-no", `${this.step + 1} / ${STEPS.length}`));
      head.append(el("h2", "tour-card-title", step.title));
      this.card.append(head);
      this.card.append(el("p", "tour-card-body", step.body));

      const chapters = el("div", "tour-chapters");
      CHAPTERS.forEach((ch) => {
        const b = el("button", "tour-chapter" + (this.step >= ch.step && (CHAPTERS[CHAPTERS.indexOf(ch) + 1]?.step ?? STEPS.length) > this.step ? " active" : ""), ch.label);
        b.type = "button";
        b.onclick = () => this.jump(ch.step);
        chapters.append(b);
      });
      this.card.append(chapters);

      const nav = el("div", "tour-nav");
      const back = el("button", "btn ghost", "Back");
      back.disabled = this.step === 0;
      back.onclick = () => this.go(-1);
      const skip = el("button", "btn ghost", "Skip to FAQ");
      skip.onclick = () => this.jump(STEPS.length - 1);
      const next = el("button", "btn primary", this.step === STEPS.length - 2 ? "FAQ" : "Next");
      next.onclick = () => this.go(1);
      nav.append(back, skip, next);
      this.card.append(nav);

      if (step.placement === "center") {
        this.card.classList.add("centered");
        this.positionCardCenter();
        return;
      }

      this.card.classList.remove("centered");
      const rect = this.resolveTarget(step);
      if (rect) {
        this.positionSpotlight(rect, step.pad || 8);
        this.positionCardNear(rect, step.placement || "below");
      } else {
        this.spot.hidden = true;
        this.positionCardCenter();
      }

      if (step.tryClick && step.target) {
        const sel = step.target;
        this.tryHandler = (e) => {
          const t = e.target.closest?.(sel);
          if (!t) return;
          e.preventDefault();
          e.stopPropagation();
          this.toast("Demo only - in a real project this would queue generation.");
          setTimeout(() => this.go(1), 400);
        };
        document.addEventListener("click", this.tryHandler, true);
      }
    },

    resolveTarget(step) {
      const nodes = [];
      if (step.target) {
        const n = document.querySelector(step.target);
        if (n) nodes.push(n);
      }
      if (!nodes.length && step.optionalTarget) {
        const n = document.querySelector(step.optionalTarget);
        if (n) nodes.push(n);
      }
      if (step.alsoHighlight) {
        const extra = document.querySelector(step.alsoHighlight);
        if (extra) nodes.push(extra);
      }
      if (!nodes.length) return null;
      let r = nodes[0].getBoundingClientRect();
      for (let i = 1; i < nodes.length; i++) {
        const ri = nodes[i].getBoundingClientRect();
        r = {
          top: Math.min(r.top, ri.top),
          left: Math.min(r.left, ri.left),
          bottom: Math.max(r.bottom, ri.bottom),
          right: Math.max(r.right, ri.right),
          width: 0,
          height: 0,
        };
      }
      r.width = r.right - r.left;
      r.height = r.bottom - r.top;
      return r;
    },

    positionSpotlight(rect, pad) {
      const t = Math.max(0, rect.top - pad);
      const l = Math.max(0, rect.left - pad);
      const w = rect.width + pad * 2;
      const h = rect.height + pad * 2;
      Object.assign(this.spot.style, {
        top: t + "px",
        left: l + "px",
        width: w + "px",
        height: h + "px",
      });
    },

    positionCardCenter() {
      Object.assign(this.card.style, { top: "50%", left: "50%", transform: "translate(-50%, -50%)", maxWidth: "520px" });
    },

    positionCardNear(rect, placement) {
      const margin = 14;
      const cardW = Math.min(420, window.innerWidth - 32);
      this.card.style.maxWidth = cardW + "px";
      let top = rect.bottom + margin;
      let left = rect.left;
      if (placement === "above") top = rect.top - margin;
      if (left + cardW > window.innerWidth - 16) left = window.innerWidth - cardW - 16;
      if (left < 16) left = 16;
      if (top > window.innerHeight - 200) top = Math.max(16, rect.top - 220);
      Object.assign(this.card.style, { top: top + "px", left: left + "px", transform: "none" });
    },

    renderFaq(step) {
      clear(this.card);
      this.card.className = "tour-card tour-faq-card centered";
      this.spot.hidden = true;
      this.root.querySelector(".tour-backdrop").classList.add("faq-mode");

      const head = el("div", "tour-card-head");
      head.append(el("span", "tour-step-no", "FAQ"));
      head.append(el("h2", "tour-card-title", step.title));
      this.card.append(head);

      const search = el("input", "tour-faq-search");
      search.type = "search";
      search.placeholder = "Search questions…";
      this.card.append(search);

      const list = el("div", "tour-faq-list");
      const renderList = (query) => {
        clear(list);
        const q = (query || "").trim().toLowerCase();
        FAQ.forEach((section) => {
          const items = section.items.filter((it) =>
            !q || it.q.toLowerCase().includes(q) || it.a.toLowerCase().includes(q));
          if (!items.length) return;
          list.append(el("div", "tour-faq-cat", section.cat));
          items.forEach((it) => {
            const det = el("details", "tour-faq-item");
            det.append(el("summary", null, it.q));
            det.append(el("p", null, it.a));
            list.append(det);
          });
        });
        if (!list.childNodes.length) {
          list.append(el("p", "tour-faq-empty", "No matches. Try another keyword."));
        }
      };
      search.oninput = () => renderList(search.value);
      renderList("");
      this.card.append(list);

      const nav = el("div", "tour-nav");
      const back = el("button", "btn ghost", "Back");
      back.onclick = () => {
        this.root.querySelector(".tour-backdrop").classList.remove("faq-mode");
        this.go(-1);
      };
      const again = el("button", "btn ghost", "Restart tour");
      again.onclick = () => {
        this.root.querySelector(".tour-backdrop").classList.remove("faq-mode");
        this.jump(0);
      };
      const done = el("button", "btn primary", "Open real editor");
      done.onclick = () => this.exit(true);
      nav.append(back, again, done);
      this.card.append(nav);
      this.positionCardCenter();
      this.card.style.maxWidth = "640px";
    },
  };

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  patchApi();
  if (window.Store) patchStore(window.Store);

  window.TourSandbox = { bootstrap, patchStore, patchApi, tourUrl, editorUrl, TOUR_KEY };
  window.TourGuide = TourGuide;
})();
