# FunPack Movie Editor (V1)

A web app for assembling LTXAV montages over the FunPack pipeline: scenes on an editable
timeline, per-scene prompts, a transition library between scenes, and one-click generation.
It runs **inside ComfyUI** — routes are registered on ComfyUI's own server, so the editor
opens at **`http://<your-comfyui-host>:<port>/funpack/movie/`** (same host/port as ComfyUI;
no separate process, no extra dependencies).

> **V1 scope.** Editable timeline + transition library feeding the **existing** uniform
> Studio → Chain Sampler chain (one generation pass). Per-scene length/FPS, Empty/Image
> scene latent building, the media browser, frame→scene anchors, true selective regen, and
> cuts/trims are later phases — their controls already exist in the UI/data model but are
> not yet wired to generation. See `/Users/dex/.claude/plans/enchanted-plotting-koala.md`.

## Architecture

```
Browser ──(same origin)──► ComfyUI server (aiohttp)
  vanilla JS frontend         /funpack/movie/         static UI
                              /funpack/movie/api/*     this module (project store, prompt assembly)
                                 ├─ in-process: parse_timeline / transitions (FunPack functions)
                                 └─ loopback HTTP: /prompt /history /view (queue + results)
```

No FastAPI/uvicorn/httpx — everything rides on ComfyUI's bundled aiohttp. The port is
detected from the running server, so non-default `--port` works automatically; the browser
uses same-origin relative URLs so it never needs to know the port at all.

## Step 0 (required for generation): export your workflow

The app queues **your** working graph. Export it from ComfyUI:

1. Build/open your Studio → Chain Sampler → VAE decode → save-video workflow.
2. Enable dev mode in ComfyUI settings, then **Save (API Format)**.
3. Save it to `movie_editor/backend/templates/ltxav_chain.api.json`.

If the app can't find the text node that feeds Studio's prompt, create
`ltxav_chain.api.json.bindings.json` next to the template, e.g.:

```json
{ "prompt": [{ "node_id": "12", "input_key": "text" }] }
```

(Find node ids via `GET /funpack/movie/api/health` then inspect the template, or read the
exported JSON.) Editing the timeline and the split preview work **without** the template —
only the Generate button needs it.

## Run

Nothing to start separately — the editor loads with ComfyUI:

```
1. Start ComfyUI normally (any --port). ComfyUI-FunPack registers the routes on load.
2. Open  http://<comfyui-host>:<port>/funpack/movie/
```

### Environment (all optional)

| Var | Default | Meaning |
|-----|---------|---------|
| `FUNPACK_MOVIE_DATA` | `~/.funpack_movie` | project store dir |
| `FUNPACK_MOVIE_TEMPLATE` | `backend/templates/ltxav_chain.api.json` | workflow template |
| `FUNPACK_COMFY_URL` | auto (detected from the running server) | override only if loopback self-calls need a different address |

## Interface ("Cutting Room")

A classic NLE three-zone layout with a top menu bar:

```
 ┌ menu bar — File · Edit · View · Generate · FunPack ─────────────┐
 ├───────────────┬────────────────────────────────────────────────┤
 │ Media Browser │  Preview (program monitor)                      │
 │  projects     │                                                 │
 │  media bin    ├────────────────────────────────────────────────┤
 │  (drag-drop)  │  Inspector — selected clip OR project + split   │
 ├───────────────┴────────────────────────────────────────────────┤
 │ Timeline — clips with transition seams; click a clip to edit it │
 └──────────────────────────────────────────────────────────────────┘
```

Editing happens in the **Inspector**: select a clip in the timeline to edit its prompt,
source (Empty/Image), transition, and exclude flag; deselect (Generate ▸ Project Settings)
to edit the anchor and global seed/frames/fps. Panel borders are draggable. The media bin
drag-drop and Render/Export are stubbed for later phases.

## Models (loaders & pipeline nodes)

The generation path is fixed (loaders → Studio → Chain Sampler → decode → save); only the
**loaders and a couple of pipeline nodes vary per machine** because they depend on what's
installed. The **Models** menu configures them model-agnostically:

- **Refresh model list** — re-scan ComfyUI (`/object_info`), same as pressing `R` in ComfyUI,
  so loader dropdowns show current files.
- **Settings…** — *Add Model / Node*: pick a **type** (Unet, LoRA, Video VAE, Audio VAE, CLIP,
  CLIP Vision, Input Image Processing, Empty Latent), then a **node** (only nodes that produce
  that type are offered). The chosen node's inputs are rendered inline — combos populated with
  your installed files — so it stays fully controlled. This "wires" the node into the pipeline.

Each configured node also gets a **Wire to** selector per output: choose where its output
connects — a **pipeline port** (Studio/Chain Sampler input, derived live from those nodes'
real `INPUT_TYPES`) or **another configured node's input** of the matching type. So you can
chain nodes (e.g. CLIP Vision loader → CLIPVisionEncode → Studio, or add an
`LTXAV Latent Combine` and wire only its output into Studio) — rebuilding the graph as a tidy
list instead of a canvas of spaghetti. Destinations are type-checked (only matching sockets
are offered).

Discovery uses `/object_info` output types; source roles exclude patchers (e.g. a LoRA loader
won't show under Unet). Config is stored globally at `~/.funpack_movie/models.json`.

> The graph builder that injects these configured loaders into the fixed path is the next step
> and needs the actual montage workflow JSON to encode exactly (the loader UI above works now).

## How the timeline maps to Studio

Studio splits one prompt into scenes by transition triggers; `segments[0]` (text before the
first trigger) is the **anchor** prepended to every scene. The app therefore emits a marker
before every scene: `intro_transition` separates the anchor from scene 1, then each scene is
preceded by the previous scene's transition. Empty markers fall back to generic `scene N`
labels. The **Split preview** panel shows exactly what Studio will see (live round-trip
through the same split logic Studio uses) and warns if the scene count doesn't match.

## Tests

```bash
pytest movie_editor/tests -q
```
