// Auto-generate a FunPack Cutting Room feature showcase — an annotated screenshot
// gallery + a walkthrough video — by driving the built-in in-app tour.
//
// Why this works with zero effort: loading the editor with `?mode=tour` boots a fully
// MOCKED demo (the tour stubs the whole API, media bin, models, shortcuts, overlays),
// so the UI renders complete with NO ComfyUI backend, NO models, NO GPU. The tour is a
// list of steps that each open the relevant panel, spotlight a target, and carry an
// authored caption — exactly the per-feature material we want. We just serve the static
// frontend, walk `window.TourGuide`, and screenshot/record each step.
//
// Self-maintaining: add a tour step → it appears in the showcase automatically.
//
//   cd tools/showcase && npm install && npx playwright install chromium && npm run capture
//
// Output (gitignored): tools/showcase/out/  — NN-*.png stills, index.md gallery,
// walkthrough.webm and (if ffmpeg is present) walkthrough.mp4.

import { chromium } from "playwright";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(__dirname, "../../movie_editor/frontend");
const OUT = path.resolve(__dirname, "out");
const VIEW = { width: 1440, height: 900 };
const SETTLE_MS = 1200; // let a step's panel/animation settle before the shot (mocked → fast)

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
  ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
  ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf", ".map": "application/json",
};

// Minimal static server for the frontend dir — no extra dependency, no ComfyUI needed.
function serve(dir) {
  const server = http.createServer((req, res) => {
    const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
    const fp = path.normalize(path.join(dir, urlPath === "/" ? "/index.html" : urlPath));
    if (!fp.startsWith(dir)) { res.writeHead(403); return res.end("forbidden"); }
    fs.readFile(fp, (err, data) => {
      if (err) { res.writeHead(404); return res.end("not found"); }
      res.writeHead(200, { "content-type": MIME[path.extname(fp).toLowerCase()] || "application/octet-stream" });
      res.end(data);
    });
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

const slug = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "step";

async function main() {
  if (!fs.existsSync(path.join(FRONTEND, "index.html"))) {
    throw new Error(`frontend not found at ${FRONTEND}`);
  }
  fs.rmSync(OUT, { recursive: true, force: true });
  fs.mkdirSync(OUT, { recursive: true });

  const server = await serve(FRONTEND);
  const { port } = server.address();
  const url = `http://127.0.0.1:${port}/index.html?mode=tour`;
  console.log(`serving ${FRONTEND} → ${url}`);

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: VIEW,
    deviceScaleFactor: 2,
    recordVideo: { dir: OUT, size: VIEW },
  });
  const page = await context.newPage();
  await page.goto(url, { waitUntil: "load" });

  // Wait for the editor shell + tour singleton to be ready.
  await page.waitForFunction(
    () => window.TourGuide && window.Store && document.getElementById("workspace"),
    null, { timeout: 30000 },
  );
  await page.waitForTimeout(1500);

  // Mount the tour and discover the step count from the card header ("1 / N").
  const count = await page.evaluate(() => {
    window.TourGuide.start(0);
    const t = document.querySelector(".tour-step-no")?.textContent || "1 / 1";
    return parseInt((t.split("/")[1] || "1").trim(), 10) || 1;
  });
  console.log(`tour has ${count} steps`);

  const steps = [];
  for (let i = 0; i < count; i++) {
    const meta = await page.evaluate((idx) => {
      window.TourGuide.jump(idx);
      return {
        title: document.querySelector(".tour-card-title")?.textContent || `Step ${idx + 1}`,
        body: document.querySelector(".tour-card-body")?.textContent || "",
      };
    }, i);
    await page.waitForTimeout(SETTLE_MS);
    const file = `${String(i + 1).padStart(2, "0")}-${slug(meta.title)}.png`;
    await page.screenshot({ path: path.join(OUT, file) });
    steps.push({ ...meta, file });
    console.log(`  ${i + 1}/${count}  ${meta.title}`);
  }

  await page.waitForTimeout(800); // tail for the video
  const video = page.video();
  await context.close();          // finalizes the recording
  await browser.close();
  server.close();

  let webm = null;
  if (video) {
    webm = path.join(OUT, "walkthrough.webm");
    try { fs.renameSync(await video.path(), webm); } catch (_) { webm = null; }
  }

  // Gallery index.
  const md = [
    "# FunPack Cutting Room — feature showcase",
    "",
    "_Auto-generated from the in-app tour by `tools/showcase/capture.mjs`. No models or GPU required — re-run after UI changes to refresh._",
    "",
    webm ? "**Walkthrough:** `walkthrough.mp4` (or `walkthrough.webm`)\n" : "",
    ...steps.map((s, i) => `## ${i + 1}. ${s.title}\n\n![${s.title}](${s.file})\n\n${s.body}\n`),
  ].join("\n");
  fs.writeFileSync(path.join(OUT, "index.md"), md);

  // Optional webm → mp4 (web-friendly) if ffmpeg is on PATH.
  if (webm) {
    try {
      execFileSync("ffmpeg", [
        "-y", "-i", webm, "-movflags", "faststart", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", path.join(OUT, "walkthrough.mp4"),
      ], { stdio: "ignore" });
      console.log("wrote walkthrough.mp4");
    } catch (_) {
      console.log("ffmpeg not found — keeping walkthrough.webm only");
    }
  }

  console.log(`\nDone — ${steps.length} stills${webm ? " + walkthrough video" : ""} in ${OUT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
