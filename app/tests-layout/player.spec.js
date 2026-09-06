// Saving a video frame, where it needs a real <video> element and a real
// canvas -- jsdom has neither. The dev server has no ComfyUI behind it (no
// /upload/image, same as no /view elsewhere), so the upload itself is
// intercepted; what is under test is that a real browser can actually draw
// the frame on screen into a real PNG blob and post it.

import { test, expect } from "@playwright/test";

// A two-frame, 32x32 red mp4 -- ffmpeg's own smallest sane output, small
// enough to inline. Real enough for Chromium to decode and report a real
// videoWidth/videoHeight, which is the one thing that matters here.
const TINY_MP4 = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAMybW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAA+gAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAlx0cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAA+gAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAACAAAAAgAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAPoAAAAAAABAAAAAAHUbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAABAAAAAQABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABf21pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAT9zdGJsAAAAv3N0c2QAAAAAAAAAAQAAAK9hdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAACAAIABIAAAASAAAAAAAAAABFUxhdmM2Mi4yOC4xMDEgbGlieDI2NAAAAAAAAAAAAAAAGP//AAAANWF2Y0MBZAAK/+EAGGdkAAqs2UlsBEAAAAMAQAAAAwEDxIllgAEABmjr48siwP34+AAAAAAQcGFzcAAAAAEAAAABAAAAFGJ0cnQAAAAAAAAW6AAAAAAAAAAYc3R0cwAAAAAAAAABAAAAAgAAIAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAABxzdHNjAAAAAAAAAAEAAAABAAAAAgAAAAEAAAAcc3RzegAAAAAAAAAAAAAAAgAAAtAAAAANAAAAFHN0Y28AAAAAAAAAAQAAA2IAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAATGF2ZjYyLjEyLjEwMQAAAAhmcmVlAAAC5W1kYXQAAAKtBgX//6ncRem95tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTY1IHIzMjIyIGIzNTYwNWEgLSBILjI2NC9NUEVHLTQgQVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDI1IC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcveDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVmPTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6b25lPTIxLDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MSBsb29rYWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFjZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJhbWlkPTIgYl9hZGFwdD0xIGJfYmlhcz0wIGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdlaWdodHA9MiBrZXlpbnQ9MjUwIGtleWludF9taW49MiBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9MjMuMCBxY29tcD0wLjYwIHFwbWluPTAgcXBtYXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAAABtliIQAFP/+7Np+BTY91+fFNDe1lLW3K4+Aln0AAAAJQZohbEEv/rXA";

async function app(page) {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
}

test("a real frame from a real video is captured and posted as an upload", async ({ page }) => {
  let posted = null;
  await page.route("**/upload/image", (route) => {
    posted = route.request().postDataBuffer();
    return route.fulfill({ json: { name: "captured.png", subfolder: "", type: "input" } });
  });

  await app(page);
  await page.evaluate((src) => {
    window.FunPack.viewer.setSource(src, "video", { filename: "clip.mp4", subfolder: "", type: "output" });
  }, TINY_MP4);

  // A real decode, not a stub: nothing here is faked once the <video> exists.
  await expect.poll(() => page.evaluate(() =>
    document.querySelector(".cx-viewer-media")?.videoWidth || 0)).toBeGreaterThan(0);

  await page.getByRole("button", { name: "Save this frame" }).click();

  await expect.poll(() => posted !== null).toBe(true);
  expect(posted.length, "no image data reached the upload").toBeGreaterThan(200);

  await expect.poll(() => page.evaluate(() => window.FunPack.bin.items.length)).toBe(1);
  await expect.poll(() => page.evaluate(() => window.FunPack.bin.items[0].file.filename))
    .toBe("captured.png");
});

test("saving a frame from a still image does nothing -- there is no frame to draw", async ({ page }) => {
  // The dev server has no /view, so this 404s and the viewer shows its own
  // "could not be loaded" state rather than an <img> -- irrelevant here: what
  // matters is that NOTHING playable is on screen, image or otherwise.
  await app(page);
  await page.evaluate(() => {
    window.FunPack.bin.absorb([{ filename: "a.png", subfolder: "", type: "output" }]);
  });

  let posted = false;
  await page.route("**/upload/image", (route) => { posted = true; return route.fulfill({ json: {} }); });

  await page.getByRole("button", { name: "Save this frame" }).click();
  await page.waitForTimeout(200);   // long enough for a wrongly-fired upload to have started

  expect(posted, "uploaded something with no video on screen").toBe(false);
  expect(await page.evaluate(() => window.FunPack.bin.items.length)).toBe(1); // still just the image
});
