# Cutting Room feature showcase generator

Auto-produces marketing material for the FunPack Cutting Room — an **annotated
screenshot gallery** + a **walkthrough video** — without screenshotting a single
button by hand.

It works by driving the editor's built-in **tour** (`?mode=tour`), which boots a
fully *mocked* demo (stubbed API, media bin, models, shortcuts, overlays). So it
runs **headless with no ComfyUI backend, no models, and no GPU**. Each tour step
opens the relevant panel, spotlights a target, and carries an authored caption —
exactly the per-feature material we want.

**Self-maintaining:** add or edit a tour step in
`movie_editor/frontend/tour.js` and re-run — the showcase updates automatically.

## Run

```bash
cd tools/showcase
npm install
npx playwright install chromium   # one-time, ~90 MB
npm run capture
```

Output lands in `tools/showcase/out/` (gitignored):

- `NN-<feature>.png` — one annotated still per tour step (1440×900 @2x)
- `index.md` — gallery: every still with its caption, ready to drop into a README/site
- `walkthrough.webm` and, if `ffmpeg` is on PATH, `walkthrough.mp4`

## Notes

- Covers the **editor/UI** ("what FunPack can do"). Output-*quality* claims
  ("what others cannot" — guide attention, anchor-as-guide, refinement keys,
  batch RLHF) need real generated clips and aren't produced here.
- Tweak viewport / timing at the top of `capture.mjs` (`VIEW`, `SETTLE_MS`).
- Requires only Node + Playwright's Chromium; the script serves the static
  frontend itself, so ComfyUI does not need to be running.
