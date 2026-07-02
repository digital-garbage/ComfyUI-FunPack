// Clean (no tour card, no spotlight dim) hero shots of the Cutting Room, using the
// same ?mode=tour mocked demo boot as capture.mjs — no ComfyUI, no models, no GPU.
// Output: out/hero-*.png. Run after capture.mjs (or standalone): `npm run capture:clean`.

import { chromium } from "playwright";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(__dirname, "../../movie_editor/frontend");
const OUT = path.resolve(__dirname, "out");
const VIEW = { width: 1440, height: 900 };

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
  ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
  ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf", ".map": "application/json",
};

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

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const server = await serve(FRONTEND);
  const { port } = server.address();
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: VIEW, deviceScaleFactor: 2 });
  await page.goto(`http://127.0.0.1:${port}/index.html?mode=tour`, { waitUntil: "load" });
  await page.waitForFunction(
    () => window.Store && document.getElementById("workspace"),
    null, { timeout: 30000 },
  );
  await page.waitForTimeout(1800);

  // If the tour auto-opened, strip its overlay — the mocked demo data stays.
  const strip = () => page.evaluate(() => document.getElementById("tour-root")?.remove());
  await strip();

  // 1. Hero: the full workspace with the first generative clip selected.
  await page.evaluate(() => {
    const first = (window.Store.get().project?.scenes || [])[0];
    if (first) window.Store.selectScene(first.id);
  });
  await page.waitForTimeout(600);
  await strip();
  await page.screenshot({ path: path.join(OUT, "hero-cutting-room.png") });
  console.log("hero-cutting-room.png");

  // 2. Composer: global prompt + shortcut/split libraries in the floating window.
  await page.evaluate(() => { if (!window.Composer.isOpen()) window.Composer.toggle(); });
  await page.waitForTimeout(700);
  await strip();
  await page.screenshot({ path: path.join(OUT, "hero-composer.png") });
  console.log("hero-composer.png");

  // 3. Engine Settings: Studio + Chain Sampler knobs without touching a graph.
  await page.evaluate(() => {
    if (window.Composer.isOpen()) window.Composer.toggle();
    window.EngineSettingsModal.open();
  });
  await page.waitForTimeout(700);
  await strip();
  await page.screenshot({ path: path.join(OUT, "hero-engine-settings.png") });
  console.log("hero-engine-settings.png");

  await browser.close();
  server.close();
  console.log(`Done — clean shots in ${OUT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
