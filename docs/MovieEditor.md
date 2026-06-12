# FunPack Cutting Room (Movie Editor)

The **Cutting Room** is FunPack's browser-based montage editor. It runs inside ComfyUI at:

`/funpack/movie`

Use it to build multi-scene projects, edit on a real NLE timeline, preview the full sequence, generate clips with FunPack Studio and the LTXAV Scene Chain Sampler, and export or stitch a final video — without hand-wiring a ComfyUI graph for every montage.

This document covers **every user-facing control**, what it does, why it exists, and what typically goes wrong.

---

## Table of contents

1. [Layout](#layout)
2. [First launch — Welcome screen](#first-launch--welcome-screen)
3. [Status bar and health](#status-bar-and-health)
4. [Menubar](#menubar)
5. [Dock panels](#dock-panels)
6. [Media Browser](#media-browser)
7. [Preview (Program Monitor)](#preview-program-monitor)
8. [Inspector (Settings panel)](#inspector-settings-panel)
9. [Timeline](#timeline)
10. [Generate and Render](#generate-and-render)
11. [Engine settings modal](#engine-settings-modal)
12. [Models and Pipeline modal](#models-and-pipeline-modal)
13. [Other modals and pickers](#other-modals-and-pickers)
14. [Keyboard shortcuts](#keyboard-shortcuts)
15. [Pipeline dependencies (first run)](#pipeline-dependencies-first-run)
16. [Tour and demo mode](#tour-and-demo-mode)
17. [Troubleshooting](#troubleshooting)

---

## Layout

The Cutting Room is a four-zone NLE shell:

| Zone | Purpose |
|------|---------|
| **Media Browser** (left) | Projects list, media bin, characters, libraries |
| **Preview** (top-right) | Program monitor, transport, minibar scrubber |
| **Inspector** (bottom-right, tab **Settings**) | Project/scene/overlay fields, global prompt |
| **Timeline** (bottom) | Clips, audio lanes, overlay lanes, ruler |

**Splitters** between zones can be dragged to resize panels. **Reset Layout** (View menu) clears saved sizes.

**Dock tabs** (Media / Preview / Settings) show or hide each zone. At least one panel must stay visible.

---

## First launch — Welcome screen

Shown when no project is loaded.

| Button | What it does | If it goes wrong |
|--------|--------------|------------------|
| **New Project** | Prompts for a name, creates a project, opens the editor | Empty name falls back to "Untitled montage". If ComfyUI is offline, project saves locally but generation will fail later. |
| **Open Latest Project** | Loads the most recently saved project | Disabled when no projects exist. Stale project list: use **Open recent** in File menu after entering. |
| **Load Project** | Import a `.json` / `.funpack_project.json` file | Invalid JSON shows an error toast. Missing media refs show broken thumbnails until you re-link assets. |
| **Welcome Tour** | Opens `?mode=tour` sandbox demo | Does not touch real projects. Exit tour to return to the normal editor. |

---

## Status bar and health

| Chip | Meaning |
|------|---------|
| **saved / unsaved / saving…** | Autosave state. Edits save on blur and after structural changes. |
| **ComfyUI live / offline** | Backend reachability. Offline: preview of renders may work; Generate/Render will fail. |
| **demo** (tour only) | Sandbox mode — no real API calls. |

**Log** button (menubar): opens ComfyUI backend log stream. Use when generation fails mysteriously. **Copy** copies the log; **✕** closes the panel.

---

## Menubar

### File

| Item | Shortcut | Action | Disabled / failure |
|------|----------|--------|-------------------|
| **New Project** | ⌘/Ctrl+N | Create project | — |
| **Open recent → …** | — | Load saved project | Row disabled if list empty |
| **Save Project File…** | — | Download project JSON | Disabled without open project |
| **Load Project File…** | — | Import JSON | Bad file → error message |
| **Import Media…** | — | Placeholder (not implemented) | Always disabled |
| **Delete Current Project** | — | Permanently delete | Confirm dialog; cannot undo |

### Edit

| Item | Shortcut | Action | Disabled when |
|------|----------|--------|---------------|
| **Undo** | ⌘/Ctrl+Z | Revert last edit | Nothing to undo |
| **Redo** | ⇧⌘/Ctrl+Z | Reapply undone edit | Nothing to redo |
| **Add Scene** | — | Append empty generative clip | No project |
| **Delete Scene** | — | Remove selected clip(s) | No selection |
| **Move Scene Left / Right** | — | Reorder clips in list | No selection |
| **Toggle Exclude** | — | Exclude clip from full Generate | No selection |

**Wrong:** Deleting a rendered clip keeps a **ghost** in preview until dismissed — it will not regenerate on the next full run.

### View

| Item | Action |
|------|--------|
| **Refresh Preview** | Reloads preview media from server |
| **Reset Layout** | Clears custom panel sizes |

### Help

| Item | Action |
|------|--------|
| **Welcome tour…** | Opens interactive tour (`?mode=tour`) |

In tour mode, Help also offers **Restart tour**, **Skip to FAQ**, **Exit tour**.

### Settings

| Item | Action | Disabled when |
|------|--------|---------------|
| **Engine settings…** | Opens Studio + Chain Sampler panels | No project, or custom/imported-only pipeline with both disabled |
| **Models…** | Pipeline wiring modal | Never |
| **Import ComfyUI Workflow…** | Workflow import wizard | No project |
| **Refresh model list** | Reloads ComfyUI node registry | Never |
| **Conditioning: …** | Pick conditioning slot | Built-in pipeline disabled |
| **Sampler: …** | Pick sampler slot | Built-in pipeline disabled |

### FunPack

| Item | Action | Notes |
|------|--------|-------|
| **Reset Studio session** | Arms/disarms session reset on next Generate | Clears Refiner memory for the project key |
| **Export refinement key…** | Save `.json` key file | Uses slot picker |
| **Import refinement key…** | Load key file | Overwrites matching key on disk |
| **Switch branch…** | Git branch switch + reload | Disabled if dirty working tree or git unavailable |
| **Update FunPack and reload** | Pull + restart | Same git guards |
| **Open ComfyUI** | Opens main ComfyUI UI in new tab | — |
| **Restart ComfyUI** | Restarts server (confirm) | Loses in-flight queue jobs |

---

## Dock panels

| Tab | Shows |
|-----|-------|
| **Media** | Media Browser |
| **Preview** | Program monitor |
| **Settings** | Inspector |

Visibility is stored in `localStorage`. Hiding Preview while generating still runs jobs; you just won't see progress until you show it again.

---

## Media Browser

### Projects

| Control | Action |
|---------|--------|
| **＋ New** | Same as File → New Project |
| Project row | Load that project |

### Tab: Media

| Control | Action | If wrong |
|---------|--------|----------|
| Drop zone / click | Upload image, video, or audio | Unsupported format rejected by server |
| **Filter** (All / Video / Audio / Images) | Filters grid | — |
| **Sort by** | Name, type, date | — |
| **Grid** (Auto, 1×–4×) | Column count | Auto scales with panel width |
| **Select** | Multi-select mode | In select mode, click toggles checkmarks |
| **Select all / Deselect all** | Bulk select | Footer toggles label |
| **⤓ Export** | Save selected/previewed file to disk | Nothing selected → disabled |
| **Remove selected (N)** | Delete from bin | Confirm; clips using that media lose anchor |
| Card click | Preview in monitor (normal mode) | — |
| Drag card | Drop on timeline clip, overlay lane, or video add | Wrong target ignored |
| **✎ Rename** | Inline rename | Enter saves, Esc cancels |
| **⤓** (card) | Export single asset | — |
| **✕** (card) | Delete asset | Confirm |

**Wrong:** Deleting media still referenced by a clip shows empty anchor until you assign new media.

### Tab: Characters

Character bible entries used by FunPack Studio when assigned to scenes.

| Control | Action |
|---------|--------|
| Search | Filter list |
| **＋ New** | Create character |
| Row click | Toggle on **selected scene**, or open editor if no scene selected |
| **✎ Edit** | Open form |
| **✕ Delete** | Remove character |

**Character form fields:**

| Field | Purpose |
|-------|---------|
| Name | Display label |
| Appearance / Body / Wardrobe | Text injected into generation |
| Always include / Never include | Hard phrase rules |
| Face / Body / Detail ref | Optional image pickers from Media bin |

**Save / Cancel** — Save persists to project; Cancel discards edits.

**Wrong:** Characters only affect generation when **FunPack Studio** is active and the character is assigned to the scene (Inspector or row click with scene selected).

### Tab: Shortcuts

Prompt **text expansions** (not keyboard shortcuts). Triggers in scene prompts expand to replacements at generation boundaries.

| Control | Action |
|---------|--------|
| **＋ Add** | Editor modal |
| **↓ Export / ↑ Import** | JSON library |
| **insert** | Append trigger to selected scene prompt | Alert if no scene selected |
| **✎ / ✕** | Edit / delete |

### Tab: Splits

**Generation split markers** — phrases that tell Studio where to split the montage into separate conditioning entries (distinct from timeline video transitions).

Same Add / Export / Import / apply / edit / delete pattern as Shortcuts.

| Editor field | Meaning |
|--------------|---------|
| Trigger | Phrase to detect |
| Placement | global / start / end / silent |

### Tab: Effects / Transitions

Read-only preset lists from the NLE library. **apply** adds the preset to the **selected clip** (requires clip selection on timeline).

---

## Preview (Program Monitor)

### Empty / idle states

| State | What you see |
|-------|--------------|
| No project | Empty reel |
| No renders yet | Hint to use Generate |
| Generating | Progress splash or step readout |

### During generation

| Control | Action |
|---------|--------|
| Progress bar | Chain sampler step progress |
| **■ Interrupt** | Cancels ComfyUI queue job | Only while queuing/running/pending |

### Media preview overlay

When you click a media bin item, preview replaces the program image until you click outside the preview element.

### Overlay compositing

Text and image overlays from timeline lanes draw on top of video. **Drag** to reposition; **double-click** overlay clip on timeline opens settings.

**Wrong:** Overlays without duration or off-screen may not appear in export — check Start/Duration in overlay inspector.

### Minibar

Segment chips show clip layout (rendered, pending, ghost, pause/gap). **Drag** any chip to scrub the playhead.

**Wrong:** If preview controls are missing, hard-reload (`Cmd/Ctrl+Shift+R`). A JavaScript error during render (often gap-segment related in older builds) prevents the transport bar from mounting.

### Transport bar

| Control | Action | Enabled when |
|---------|--------|--------------|
| **⏹ Stop** | Stop playback, reset to clip start | Always |
| **▶ / ⏸** | Play / pause sequence | Always |
| Timecode | Current / total position | — |
| **Anchor scene** dropdown | Pick scene for frame capture | Playhead on a rendered segment |
| **📌 Save frame** | PNG → Media bin | Rendered segment under playhead |
| **Use as anchor** | Sets scene source to `generated_frame` | Same |

**Wrong:** Save frame on a pending (unrendered) clip is disabled. Use Generate first.

---

## Inspector (Settings panel)

Toggle **Project** vs **Scene** at the top. When an overlay is selected, overlay fields replace scene fields.

### Global prompt (always visible)

Single textarea holding the **combined montage text**. Edits debounce and split back to per-scene prompts using split markers.

**Wrong:** Editing global prompt while scenes have custom per-clip text can resync scene text — check **Split preview** section if prompts drift.

### Engine strip

| Control | Action |
|---------|--------|
| Status text | Shows active Studio / Chain / custom workflow |
| **Engine settings →** | Opens engine modal |
| **Models →** | Opens Models modal (custom pipeline) |

### Project mode (no scene selected)

**Output**

| Field | Maps to | Wrong if |
|-------|---------|----------|
| Project name | Display + filenames | — |
| Frames / scene | Default clip length | Too low → very short clips |
| FPS | Frame rate | Mismatch with model training → judder |

**Prompt**

| Field | Purpose |
|-------|---------|
| Anchor | Global scene anchor phrase |
| Split before scene 1 | Intro split marker |
| Negative prompt | Passed to Studio encoding |

**Advanced**

| Field | Purpose |
|-------|---------|
| Max scenes | Cap on clip count |
| Width / Height | Output resolution for render |

### Scene mode — generative clip

| Field | Purpose | Common mistakes |
|-------|---------|-----------------|
| **Prompt** | Scene text | Empty → generic or carry-only behavior |
| **Source** | How i2v starts | Wrong source → black frame or wrong anchor |
| Source options | See table below | |
| **Generate this scene** | Queue single-scene run | Blocked if pipeline requirements missing |

**Source modes**

| Mode | Meaning | When it fails |
|------|---------|---------------|
| empty | Text-only / carry | No image anchor unless carry guides exist |
| image | Img2Video from picked media | No media assigned → generation blocked |
| generated_frame | Uses captured frame | No capture yet → blocked |
| v2v | Video-to-video from bin video | Wrong media type |
| carry | Prior-scene guides only | First scene cannot carry |
| mixed | Anchor image + prior guides | Needs anchor + Chain Sampler |

**Outgoing seam**

| Field | Purpose |
|-------|---------|
| Video transition | Dissolve, fade black, wipes between clips |
| Blend length (frames) | Transition duration (disabled for hard cut) |
| Generation split marker | Prompt marker before next scene (Studio split) |
| Pause before next clip (s) | Black/silent gap after this clip |

**More editing** (fold)

| Field | Purpose |
|-------|---------|
| Blur / Fade in/out | Post FX on render |
| Ken Burns | Zoom in/out with ramp |
| Frames / FPS modes | project / timeline / custom length |
| Source in / dur | Trim into source video |
| i2v guides | Per-guide stack (Chain Sampler) |
| Exclude from full generation | Skip on **▶ Generate** whole montage |

**Characters** — chips from Characters bin; ✕ removes from scene.

### Scene mode — video clip (locked)

Read-only source name. Same outgoing seam and basic FX. Use toolbar **Convert to scene** to make it generative again.

### Overlay inspector

**Text:** Text, Font size, Font, Color, Start, Duration, Opacity, Flip H/V, layer order, **Remove overlay**.

**Image:** Size (via pixel fields), same timing/opacity/layer/remove.

### Split preview (fold)

Shows how global prompt parses into scenes. **↺ Sync scenes from preview** rebuilds timeline scenes from parse (confirm) — destructive to manual scene list if parse differs.

### Exposed controls

Fields marked **◉ expose** in Models modal appear here dynamically (combo, number, boolean, text from wired nodes).

---

## Timeline

### Header — Generate / Render

See [Generate and Render](#generate-and-render).

### Toolbar

| Control | Action | Disabled when |
|---------|--------|---------------|
| Timecode | Playhead / total duration | — |
| **＋ Add** | Dropdown menu | No project |
| **Split** | Split clip or audio at playhead | No clip/audio selection |
| **Remove** | Delete selection | No selection |
| Selection badge | Shows multi-select count | Hidden if 0 |
| **⤓ Export** | Export rendered clip file(s) | Nothing saveable selected |
| **Save to media bin** | Copy render into bin | Same |
| **⊟ Separate audio** | Detach embedded audio to lane | No embedded audio |
| **Remove audio** | Delete separated track | No separated track |
| **Convert to scene / video** | Toggle clip type | Exactly one clip selected |
| **Rate scene…** | Refiner rating picker | Studio off or no render |
| Zoom **− / ＋ / fit** | Timeline zoom | — |

### ＋ Add menu

| Item | Creates |
|------|---------|
| Clip | Empty generative scene |
| Video | Locked video clip from bin |
| Effects | NLE effect on selected clip |
| Transitions | Video transition on selected clip |
| Text | Text overlay |
| Image | Image overlay |
| Overlay track | New overlay lane |
| Audio | Inserted audio track from bin |

### Video lane — clip interactions

| Interaction | Result |
|-------------|--------|
| Click clip | Select (Shift range, ⌘/Ctrl toggle) |
| Drag clip **horizontally** | **Add/remove pause** before clip (drag right = gap, flush left = no gap) |
| Left trim handle | Shorten from start; recomputes frames |
| Right trim handle | Shorten/lengthen end |
| Alt + left trim (rendered) | Slip source media |
| Drag colored **tail** | Video transition length |
| Drop video from bin | New video clip |
| **Remove** (clip head) | Delete clip → may leave ghost |
| Drop media on clip | Assign i2v anchor |

**Wrong trim:** Custom frames mode locks trim handles — switch Frames to "Inherit project" in Inspector.

**Wrong spacing:** First clip cannot be dragged for spacing (no prior clip). Use **Move Scene Left/Right** in Edit menu to reorder.

### Ghost clips

Removed scenes that still have renders show as ghosts (preview only). **Remove** on ghost dismisses from preview. They do **not** run on next Generate.

### Original audio lane

Per-scene embedded audio from video/render. **Volume** slider 0–2.

### Inserted audio tracks

| Control | Notes |
|---------|-------|
| Trim handles | Trim duration; Alt+left = slip |
| Volume | 0–2 |
| Start (s) | Disabled for **separated** (linked) tracks — follows video |
| Drag block | Move start time (non-separated) |
| ✕ | Remove track |

### Overlay lanes

| Control | Action |
|---------|--------|
| Gutter **✕** | Remove lane (hidden if only one) |
| Drop image | Assign to lane |
| Double-click clip | Settings modal |
| Trim / drag | Time and layer |
| ✕ on clip | Remove overlay |

### Ruler

**Drag** to scrub playhead (smooth, no full page reload).

### Meta line

Clip counts, ghosts, selection, total duration. **Notice ✕** dismisses transient messages (e.g. save confirmations).

---

## Generate and Render

| Button | What it does | Blocked when |
|--------|--------------|--------------|
| **▶ Generate** | Queue full montage (non-excluded scenes) | No project; job already running; pipeline requirements unmet |
| **Selected (N)** | Generate only selected generative scenes | No selection; busy |
| **⧉ Render** | FFmpeg stitch of rendered clips + gaps + overlays/audio | Nothing to stitch; busy |

**Generate flow**

1. Builds ComfyUI workflow from Models config + Engine settings.
2. Queues on ComfyUI.
3. Streams progress to preview; partial scene previews may appear mid-run.
4. Writes renders to ephemeral project storage; maps to timeline clips.

**Wrong:** Missing node packs → pipeline setup modal. Missing i2v anchor → friendly blocker message naming the scene. Tunnel/timeout → retry; check Log.

**Render flow**

1. Confirms clip count or overlay-only duration.
2. Stitches with transitions, gaps, FX, audio mix.
3. Offers download; may replace per-scene renders with single stitched source.

**Wrong:** Unrendered clips skipped. Overlay-only render uses blank canvas at project resolution.

---

## Engine settings modal

Open via **Settings → Engine settings** or Inspector **Engine settings →**. Escape closes.

### Preset bar

| Preset | Intent |
|--------|--------|
| Fast draft | Lower steps, lighter continuity |
| Quality | Higher quality defaults |
| Continuity-heavy | Stronger auto continuity |

Presets patch many fields at once — review before Generate if you have custom tuning.

### Pipeline summary

Shows active **Conditioning** and **Sampler** slots (built-in Studio + Chain or custom).

### FunPack Studio card

Essentials, Refinement (mode, advisor, value guidance, steer mode absolute/relative), and **Sampler algorithm** sub-panel (Hybrid / Distilled / Normalizing / KSampler parameters).

Hidden when `disable_core` or Studio slot not active.

### Chain Sampler card

Auto continuity, timing, manual continuity overrides, guidance, decode noise, experimental flags.

**Wrong:** Auto continuity off with carry-only first scene → weak motion. Identity pin without auto continuity has no effect.

### Models & Pipeline… link

Jumps to Models modal.

---

## Models and Pipeline modal

### Requirements checklist

Lists required roles (MODEL, CLIP, VAE, etc.). **+ Add** on missing row opens node picker.

**Wrong:** Red items block Generate with a message naming the unfed input.

### Built-in pipeline

Default LTX montage graph (needs ComfyUI-LTXVideo, Video Helper Suite, KJNodes — see [Pipeline dependencies](#pipeline-dependencies-first-run)).

| Control | Action |
|---------|--------|
| Expand/collapse | Show core node wiring |
| **Full control / Guided wiring** | Guided locks critical paths; Full allows rewiring |
| **Disable built-in pipeline** | Use imported workflow only |

### Add Model / Node

Pick node class from ComfyUI registry → **Add** to slot list.

### Per-slot card

Rename, remove, widget values, **◉ expose** to Inspector, wire outputs, set input sources (link to another slot's output).

### Link inputs mode

Multi-select compatible inputs → **Save link** creates one exposed control driving many nodes.

### Import ComfyUI Workflow

Four-step wizard: load JSON → review nodes → bind editor inputs → apply.

**Wrong:** Unbound required inputs block generation until wired or defaulted.

---

## Other modals and pickers

### Pipeline setup (automatic)

Prompts when built-in pipeline nodes missing. See [Pipeline dependencies](#pipeline-dependencies-first-run).

### Rating picker

Studio-style ratings for rendered scene. Categories: Positive / Missing / Wrong, heart modifier, Nuclear (Awful, Forget). Ratings train Refiner memory on next compatible run.

### Slot picker

Used for new project name, slot selection, refinement key export path, branch name.

### Overlay add/edit modals

Text/image fields as in Inspector. **Add** / **Save** commits; **Cancel** discards.

### NLE effect / transition picker

Preset dropdown, optional parameter, **Apply**.

### Video / Audio add modals

Pick from Media bin or upload.

### Restart / update overlays

Shared dimmed progress UI for ComfyUI restart and FunPack git update. **Cancel install** aborts stalled Manager queue.

---

## Keyboard shortcuts

### Global (menubar)

| Shortcut | Action |
|----------|--------|
| Escape | Close open menu |
| ⌘/Ctrl+N | New project |
| ⌘/Ctrl+Z | Undo |
| ⇧⌘/Ctrl+Z | Redo |

Blocked when focus is in input/textarea.

### Timeline

| Key | Action |
|-----|--------|
| J | Seek −1 s |
| K | Pause |
| L | Play |
| Space | Toggle play/pause |
| + / = | Zoom in |
| - / _ | Zoom out |
| ← / → | Seek ±1 frame |
| Delete / Backspace | Remove selection |
| I | Seek to selection in-point |
| O | Seek to selection out-point |
| S | Split at playhead |

Also blocked in text fields.

---

## Pipeline dependencies (first run)

When you open a project using the **built-in pipeline**, the editor checks ComfyUI for required node classes.

| Step | What happens |
|------|--------------|
| No ComfyUI-Manager | Offers to clone Manager into `custom_nodes` |
| Manager not loaded | Offers **Restart ComfyUI** |
| Missing packs | Offers install of LTXVideo, Video Helper Suite, KJNodes via Manager queue |
| **No, I'll use my own pipeline** | Sets `disable_core`; dismisses future prompts (localStorage) |
| **Close** | Dismisses modal without installing |

**Wrong:** Install stuck → **Cancel install**. Pip security policy errors → install packs manually in terminal. After Manager install, restart before pack install.

---

## Tour and demo mode

URL: `?mode=tour`

Interactive walkthrough with sandbox project. **Generate** shows toast instead of queuing. **Exit tour** returns to normal editor.

Help → **Welcome tour…** or Welcome screen **Welcome Tour**.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Preview controls missing | JS error during player render | Hard-reload `Cmd/Ctrl+Shift+R`; update FunPack |
| Generate blocked message | Unfed pipeline input or missing anchor | Models checklist; assign image/capture |
| ComfyUI offline chip | Server down or wrong URL | Start ComfyUI; open from same host |
| Clips won't play mid-sequence | Missing render for that scene | Generate scene or full montage |
| Audio out of sync | Separated track start drift | Re-separate or nudge track; trim video |
| Global prompt reverts | Autosave race or Sync from preview | Apply global prompt; avoid sync unless intended |
| Ghost clips after delete | By design for preview | Dismiss ghost or regenerate |
| Pause won't remove | Gap not flush | Drag clip left until it touches previous clip |
| Overlays missing in export | Zero duration or wrong lane | Check Start/Duration; render again |
| `TypeError: sc is undefined` (old) | Gap segment in minibar | Update to 3.0.0+ |
| Manager install fails | No git/network | Install packs manually |
| Studio memory not learning | Reset session armed; wrong key | Disarm reset; check refinement key |

### Recommended workflow

1. **New Project** or open recent.
2. Install pipeline deps if prompted.
3. Set **Engine settings** preset or tune Studio/Chain.
4. Write montage in **Global prompt** or per-scene prompts.
5. Assign **Characters** and **media anchors** where needed.
6. **▶ Generate** (full or selected).
7. Edit timeline: trim, split, spacing, transitions, overlays, audio.
8. **Rate** clips to train Refiner for next pass.
9. **⧉ Render** final stitch or **Export** individual clips.

---

## Related documentation

- [`FunPackStudio.md`](FunPackStudio.md) — Studio popup tabs and Refiner behavior (also exposed in Engine settings).
- [`FunPackLTXAVSceneChainSampler.md`](FunPackLTXAVSceneChainSampler.md) — Chain sampler parameters mirrored in Engine settings.
- [`FunPackVideoRefinerV2.md`](FunPackVideoRefinerV2.md) — Refiner V2 learning model behind Studio ratings.
