# FunPack Movie Editor (V1)

A standalone web app for assembling LTXAV montages over the FunPack pipeline: scenes on
an editable timeline, per-scene prompts, a transition library between scenes, and one-click
generation. ComfyUI is the inference backend — the app drives it over its HTTP API and
reuses FunPack's `/funpack/*` routes for parsing.

> **V1 scope.** Editable timeline + transition library feeding the **existing** uniform
> Studio → Chain Sampler chain (one generation pass). Per-scene length/FPS, Empty/Image
> scene latent building, the media browser, frame→scene anchors, true selective regen, and
> cuts/trims are later phases — their controls already exist in the UI/data model but are
> not yet wired to generation. See `/Users/dex/.claude/plans/enchanted-plotting-koala.md`.

## Architecture

```
Browser (vanilla JS)  →  FastAPI sidecar (light, no torch)  →  ComfyUI
                            project store + prompt assembly       /funpack/*  (parse, libraries)
                                                                  /prompt /history /view  (generate)
```

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

(Find node ids with `GET /api/health` then inspect the template, or read the exported JSON.)
Editing the timeline and the split preview work **without** the template — only the
Generate button needs it.

## Run

```bash
# 1. Start ComfyUI (with ComfyUI-FunPack installed) on the GPU box, e.g. :8188
# 2. Start the sidecar:
pip install -r movie_editor/backend/requirements.txt
FUNPACK_COMFY_URL=http://127.0.0.1:8188 \
  python -m movie_editor.backend.app
# 3. Open http://127.0.0.1:8200
```

### Environment

| Var | Default | Meaning |
|-----|---------|---------|
| `FUNPACK_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI base URL |
| `FUNPACK_MOVIE_HOST` / `_PORT` | `127.0.0.1` / `8200` | sidecar bind |
| `FUNPACK_MOVIE_DATA` | `~/.funpack_movie` | project store dir |
| `FUNPACK_MOVIE_TEMPLATE` | `backend/templates/ltxav_chain.api.json` | workflow template |
| `FUNPACK_MOVIE_CORS` | `*` | CORS origins |

## How the timeline maps to Studio

Studio splits one prompt into scenes by transition triggers; `segments[0]` (text before the
first trigger) is the **anchor** prepended to every scene. The app therefore emits a marker
before every scene: `intro_transition` separates the anchor from scene 1, then each scene is
preceded by the previous scene's transition. Empty markers fall back to generic `scene N`
labels. The **Split preview** panel shows exactly what Studio will see (live round-trip
through `/funpack/parse_timeline`) and warns if the scene count doesn't match.

## Tests

```bash
pytest movie_editor/tests -q
```
