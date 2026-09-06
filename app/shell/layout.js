// The app's regions, and the mount points each one offers.
//
// A region declares what it can host. Core never learns these names, and no
// list of them exists anywhere else -- a module asks for one by name, and gets
// it or does not.
//
// The shape is the one v4 called Simple mode, built the other way round. v4's
// Simple mode was the Editor with about ten `display: none !important` rules
// over it, which is not a smaller app but the same app wearing a mask -- and it
// broke: a collapsed column was marked [hidden], and nothing can slide a
// display:none element into view, so a panel button did nothing at all and said
// nothing. So here a region that does not exist yet is ABSENT. The timeline,
// the picker wheel and the project wizard are not hidden below; they have not
// been built, and adding one is adding a region rather than removing a rule.
// The bin arrived exactly that way.

import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";
import { createTransport } from "./transport.js";
import { createBin } from "./bin.js";
import { createMenubar } from "./menubar.js";
import { createConstructor } from "./constructor.js";

export function build(root, handlers = {}) {
  // The transport: Generate, Cancel, and what the run is doing. Built before the
  // zones because its controls are handed to the zone that holds them.
  const transport = createTransport(handlers);

  // Built here rather than on first open: a mount point has to exist when
  // modules mount, which is long before anyone opens a window.
  // onChange, so the Constructor toggle is right when the window closes ITSELF --
  // its own Done button, Escape, or a click on the backdrop.
  const constructor = createConstructor({ onChange: () => syncRegions() });

  // Centre: the result, and the timeline under it.
  const viewer = composer.viewer.media({ empty: "The result of the last run appears here." });

  // Left: what a project is made of, which for now is what it has produced. The
  // bin owns the choice of what the viewer shows -- one place decides, so a
  // click on an older result and a run finishing cannot both be right at once.
  const bin = createBin({
    onOpen: (item) => viewer.setSource(item.url, item.kind, item.file),
  });
  const assets = composer.panel.zone({
    title: "Assets",
    actions: [bin.control],
    body: bin.host,
  });

  // ComfyUI's own route, not FunPack's -- ordinary image upload already exists
  // and there is nothing about a captured frame that needs its own.
  async function uploadFrame(blob) {
    const form = new FormData();
    form.append("image", blob, `frame-${Date.now()}.png`);
    form.append("type", "input");
    const res = await fetch("/upload/image", { method: "POST", body: form });
    if (!res.ok) throw new Error(`the upload was refused (${res.status})`);
    const body = await res.json();
    return { filename: body.name, subfolder: body.subfolder || "", type: body.type || "input" };
  }

  // `transport.say` is the Timeline's own line, overwritten by the next
  // progress message a run in flight sends -- often inside a second, which
  // silently erased a save's own confirmation or error whenever anything was
  // generating. This one is the Preview zone's, and nothing else writes to it.
  const previewStatus = composer.text.sm({ text: "" });
  let previewStatusTimer = null;
  function sayPreview(text, ms = 3000) {
    previewStatus.setText(text);
    if (previewStatusTimer) clearTimeout(previewStatusTimer);
    previewStatusTimer = setTimeout(() => previewStatus.setText(""), ms);
    // Not a reason to keep a process alive: under a test runner, a page built
    // and abandoned mid-test (a new one built for the next test, this one's
    // DOM never explicitly torn down) left a timer that fired minutes later,
    // reaching for a `document` the test harness had already removed.
    if (previewStatusTimer && typeof previewStatusTimer.unref === "function") previewStatusTimer.unref();
  }

  // What's shown here came from somewhere already server-side (a run's own
  // result, or an already-open bin/temp item) OR is a video the viewer can
  // still draw a frame from -- either way, "keep this" should not mean a
  // download and a re-upload by hand.
  const saveToBin = composer.iconButton.sm({
    icon: "➕", label: "Save to bin",
    onClick: () => {
      if (!viewer.file) { sayPreview("Nothing here has a file to save."); return; }
      // Already showing, so nothing needs to move -- but a save is a save,
      // never a navigation, even when it happens to name what's on screen.
      const added = bin.absorb([viewer.file], { open: false });
      sayPreview(added.length ? "Saved to the bin." : "Already in the bin.");
    },
  });
  const saveFrame = composer.iconButton.sm({
    icon: "⛶", label: "Save this frame",
    onClick: async () => {
      const blob = await viewer.captureFrame();
      if (!blob) { sayPreview("Play a video first -- there is no frame to save."); return; }
      try {
        // Not `open: true`: the upload is a real round trip, and by the time
        // it lands the viewer may be showing something else entirely -- a
        // save must not yank it back to what it saved.
        const added = bin.absorb([await uploadFrame(blob)], { open: false });
        sayPreview(added.length ? "Frame saved to the bin." : "Already in the bin.");
      } catch (err) {
        sayPreview(`Could not save the frame: ${err.message}`);
      }
    },
  });
  // v4's preview head carries nothing but the name of the zone: what is being
  // looked at is not something you act on -- except now it is, the two
  // actions above, which only ever touch what is ALREADY on screen.
  const preview = composer.panel.zone({
    title: "Preview", body: viewer, flush: true,
    actions: [saveToBin, saveFrame], status: [previewStatus],
  });

  // The timeline: what the project IS, and where a run is started from.
  //
  // Generate lives here rather than in a bar across the bottom, because a bar
  // that spans the window belongs to no region and acts on whatever happens to
  // be in front -- which is a dashboard. v4 puts Generate in the timeline head,
  // beside the thing it fills, and so does this.
  //
  // The warning is the first thing INSIDE the zone rather than in its head: it
  // is a sentence, not a chip, and it has to sit under the button it is about
  // -- an inert setting is said where the run is started, not where it was
  // switched on.
  // The zone's body is held rather than built inline: boot replaces the stand-in
  // with the real timeline once there is a project, and layout stays ignorant of
  // what a scene is. The warning stays put -- it is about the run, not the list.
  const timelineEmpty = composer.emptyState.default({
    icon: "▭",
    title: "Nothing on the timeline",
    hint: "What you generate lands here, in the order it plays.",
  });
  // The region toggles, v4's arrangement: named buttons, grouped, in the bar
  // above the timeline. The kit's own fallback is a glyph on a rail at each
  // outer edge of the window -- fine for a kit that cannot know what its panels
  // are called, wrong here, where they have names and a bar to put them in. A
  // control whose meaning has to be discovered is not a control.
  //
  // They are toggles and say so (aria-pressed), and the workspace stays the one
  // source of truth for what is open: these ask it to toggle and are told back
  // through onToggle, so a panel closed by a narrow window updates its button.
  let ws = null;
  const regionButtons = [];
  const syncRegions = () => {
    // Too narrow to dock means the panel OVERLAYS the centre -- and this button
    // is in the centre, so the panel it opens covers it. A control that can be
    // hidden by the thing it operates is not a control: it goes away, and the
    // View menu in the bar, which nothing can cover, is the way in and out.
    const covered = Boolean(ws && ws.narrow());
    for (const r of regionButtons) {
      r.button.setPressed(r.isOn());
      if (r.overlaid) r.button.node.toggleAttribute("hidden", covered);
    }
  };
  const regionToggle = (label, isOn, act, { overlaid = false } = {}) => {
    const button = composer.button.sm({ label, pressed: isOn(), onClick: () => { act(); syncRegions(); } });
    regionButtons.push({ button, isOn, overlaid });
    return button;
  };

  const timelineBody = composer.region.stack({ gap: "sm", fill: true,
                                              children: [transport.warning, timelineEmpty] });
  const timeline = composer.panel.zone({
    title: "Timeline",
    actions: [
      ...transport.actions,
      // A ready-made widget, not a callback: the loop it runs needs the run,
      // the session and the project all at once, and none of those exist yet
      // at the point build() is called -- boot.js is the one place all three
      // do, so boot.js builds the button too.
      ...(handlers.generateAll ? [handlers.generateAll] : []),
      regionToggle("Assets", () => Boolean(ws && ws.isOpen("left")),
                   () => ws && ws.toggle("left"), { overlaid: true }),
      regionToggle("Properties", () => Boolean(ws && ws.isOpen("right")),
                   () => ws && ws.toggle("right"), { overlaid: true }),
      regionToggle("Constructor", () => constructor.isOpen, () => {
        if (constructor.isOpen) constructor.close();
        else if (handlers.onConstructor) handlers.onConstructor();
      }),
    ],
    status: transport.status,
    body: timelineBody,
  });

  // The stand-ins below are EMPTY STATES, not labels: they say a region is
  // empty, and settle() takes them down once it is not. A line reading "modules
  // appear here" left above the modules that appeared is a region explaining
  // itself to nobody.

  // The result gets the room. Draggable, so a long timeline can take it back.
  const centre = composer.splitPane.v({
    size: 74,
    label: "Preview and timeline",
    panes: [preview, timeline],
  });

  // Right: everything about the run. Generation is open, settings folded away
  // -- one is read constantly and the other rarely, and a fold says which is
  // which without needing a second place to look. A settings WINDOW is a
  // separate thing that has not been built; this is not a stand-in for it.
  const generationEmpty = composer.emptyState.default({
    icon: "◎",
    title: "Nothing to set",
    hint: "Modules with settings for a run appear here.",
  });
  const generation = composer.region.stack({ gap: "sm", label: "Generation",
                                            children: [generationEmpty] });

  const settingsEmpty = composer.emptyState.default({
    icon: "⚙",
    title: "Nothing to set",
    hint: "Modules with settings that are not about one run appear here.",
  });
  // A host, not a panel: a bordered panel inside the collapsible's own bordered
  // box is two frames drawn around one list of settings.
  const settings = composer.region.stack({ gap: "sm", label: "Settings",
                                           children: [settingsEmpty] });
  // ONE zone, not two panels stacked. A region of the app is one area with one
  // head on it; two heads in a column means two things, and there is only one
  // thing here -- what the next run will do.
  // The inspector goes ABOVE the module panels: what this scene and this project
  // are is read more often than any switch, and a column is read downwards.
  const propertyRows = composer.region.stack({
    gap: "sm", label: "Properties",
    children: [
      // What this scene and this project ARE, above the switches: a column is
      // read downwards, and this is the half that is read every time.
      handlers.inspector,
      generation,
      composer.collapsible.default({ label: "Settings", body: settings }),
    ].filter(Boolean),
  });
  // Titled after what it is ABOUT, and it follows the timeline -- v4's right
  // column reads "Scene · 1" and so does this. boot renames it on selection.
  const properties = composer.panel.zone({ title: "Scene", body: propertyRows });

  const workspace = composer.workspace.docked({
    id: "main",
    leftLabel: "Assets",
    rightLabel: "Properties",
    left: assets,
    centre,
    right: properties,
    // Named toggles live in the timeline bar instead -- see regionToggle above.
    rails: false,
    onToggle: () => syncRegions(),
  });
  ws = workspace;
  syncRegions();


  // The menu bar: who this is, what is not a zone, and whether ComfyUI is
  // still on the other end. A zone head holds what acts on THAT zone, so
  // anything that acts on the app has nowhere else to live.
  const menubar = createMenubar({
    workspace,
    onPipeline: handlers.onPipeline,
    onUpdates: handlers.onUpdates,
    onPacks: handlers.onPacks,
    onLog: handlers.onLog,
    onTemp: handlers.onTemp,
    onProject: handlers.onProject,
    edits: handlers.edits,
    projects: handlers.projects,
    current: handlers.currentProject,
  });
  const bar = menubar;

  const page = composer.frame.app({ header: bar, main: workspace });

  root.replaceChildren(page.node);

  // The mount point IS the contract with modules; the region behind it can be
  // rearranged freely as long as the name survives.
  offer("assets.library", assets.body);
  // The prompt is written in a WINDOW, not in a zone: it is the longest thing a
  // person writes and the least often read, and a permanent third of the centre
  // column is the room the timeline needs.
  offer("generation.prompt", constructor.written.node, constructor.empty.node);
  // What the project generates at. A pipeline with no such inputs offers no
  // roles, nothing mounts, and the group is simply not there.
  offer("project.video", constructor.video.node);
  offer("settings.general", settings.node, settingsEmpty.node);
  // One host, five names. Each carries the same stand-in, and settle() takes it
  // down once anything at all has mounted into the panel they share.
  for (const point of ["model", "latent", "sampling", "timing", "post"]) {
    offer(`generation.${point}`, generation.node, generationEmpty.node);
  }

  return { workspace, assets, bin, preview, viewer, timeline, timelineBody,
           timelineEmpty, generation, settings, syncRegions,
           properties, transport, constructor, menubar,
           connection: menubar.connection, bar, page };
}
