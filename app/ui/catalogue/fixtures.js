// Hide, don't warn -- shown rather than asserted.
//
// One good module and five broken ones, put through the SAME shell code the app
// uses. The good one's panel appears. The broken ones do not appear at all: no
// placeholder, no greyed row, no warning chip. What you see on this page is the
// whole of what a user would see.
//
// These mirror core/tests/fixtures/fixture_modules, so the browser and the
// Python suite are looking at the same five ways of being wrong.

export const GOOD = {
  id: "audio_clock",
  title: "Audio clock",
  mount: "demo.panel",
  stage: "conditioning",
  status: "proven",
  requires: ["audio_stream"],
  settings: {
    enabled: { type: "bool", default: true, label: "Sync to audio clock",
               hint: "Locks frame timing to the audio stream." },
    strength: { type: "float", default: 0.65, min: 0, max: 1, step: 0.05,
                label: "Lock strength", unit: "x", ui: "slider", when: { enabled: true } },
    mode: { type: "enum", default: "beat", label: "Alignment", ui: "segmented",
            options: [{ value: "beat", label: "Beat grid" },
                      { value: "onset", label: "Onset" },
                      { value: "flat", label: "Flat" }],
            when: { enabled: true } },
  },
};

// Each of these is refused somewhere different, which is the point: they are
// caught at the earliest layer that can catch them.
export const BROKEN = [
  {
    why: "raises on import",
    caughtBy: "core: the scan imports it in a try",
    spec: { id: "explodes", title: "Explodes", mount: "demo.panel", settings: {}, ui: "/nope/missing.js" },
  },
  {
    why: "asks for a renderer its type does not have",
    caughtBy: "core: schema validation, before the manifest",
    spec: { id: "bad_ui", title: "Bad renderer", mount: "demo.panel",
            settings: { on: { type: "bool", default: true, label: "On", ui: "hologram" } } },
  },
  {
    why: "a label that is not text",
    caughtBy: "core: schema validation; the kit would refuse it too",
    spec: { id: "bad_label", title: "Bad label", mount: "demo.panel",
            settings: { on: { type: "bool", default: true, label: { markup: "<b>no</b>" } } } },
  },
  {
    why: "no default, so a headless run has no value",
    caughtBy: "core: schema validation",
    spec: { id: "no_default", title: "No default", mount: "demo.panel",
            settings: { strength: { type: "float", min: 0, max: 1, label: "Strength" } } },
  },
  {
    why: "names a mount point no region offers",
    caughtBy: "shell: mounts.js has no host for it",
    spec: { id: "lost", title: "Lost module", mount: "somewhere.else",
            settings: { on: { type: "bool", default: true, label: "On" } } },
  },
];
