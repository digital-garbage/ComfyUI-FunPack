// About FunPack: which build this is, and the machine ComfyUI runs on (which on a rental is not this browser's).
import { composer as c } from "../../composer/composer.js";

const facts = (rows) => rows.filter(([, v]) => v).map(([k, v]) => c.settingsRow.default({ label: k, control: c.text.sm({ text: String(v) }) }));

const hero = (name, sub) => {
  const node = c.region.stack({ gap: "none", children: [c.brand.default({ name: "" }), c.header.xl({ text: name }), c.hint.default({ text: sub })] });
  node.node.classList.add("cx-settings-hero");
  return node;
};

export const about = (api) => function mount() {
  const body = c.region.stack({ gap: "sm", label: "About" });
  body.set([c.hint.default({ text: "Looking…" })]);
  Promise.all([api.gitStatus().catch(() => ({})), api.system().catch(() => null)]).then(([git, sys]) => {
    const major = String(git.version || "").split(".")[0];
    const mem = (sys && sys.memory) || {}, cpu = (sys && sys.cpu) || {}, disk = (sys && sys.disk) || {}, torch = (sys && sys.torch) || {}, gpu = ((sys && sys.gpus) || [])[0];
    body.set([
      hero(`FunPack${major ? ` ${major}` : ""}`, [git.codename && `“${git.codename}”`, "Cutting Room"].filter(Boolean).join(" · ")),
      ...facts([["Version", git.version], ["Commit", git.commit && String(git.commit).slice(0, 7)], ["Branch", git.branch]]),
      ...(sys ? [c.label.section({ text: "Hardware" }),
        ...facts([["Chip", cpu.name], ["Memory", mem.total_gb != null && `${mem.available_gb} GB free of ${mem.total_gb} GB`],
          ["Graphics", gpu ? [gpu.name, gpu.vram_gb && `${gpu.vram_gb} GB`].filter(Boolean).join(" · ") : sys.mps ? "Apple GPU (MPS)" : "CPU only"],
          ["Storage", disk.total_gb != null && `${disk.free_gb} GB available of ${disk.total_gb} GB`]]),
        c.label.section({ text: "Software" }),
        ...facts([["System", sys.os], ["ComfyUI", sys.comfyui], ["Python", sys.python], ["Torch", [torch.version, torch.cuda && `CUDA ${torch.cuda}`].filter(Boolean).join(" · ")], ["Host", sys.host]]),
        c.hint.default({ text: "The machine ComfyUI runs on — not this browser." })] : [c.hint.default({ text: "Machine details are not available from this server." })]),
    ]);
  });
  return body;
};

/** "Is this machine ready to generate?": the rows core/readiness.py reports, worst first. */
export const readiness = (api) => function mount() {
  const body = c.region.stack({ gap: "sm", label: "Ready to generate" });
  const MARK = { ok: "✓", warn: "!", fail: "✗" }, TONE = { ok: "good", warn: "warn", fail: "danger" }, ORDER = { fail: 0, warn: 1, ok: 2 };
  const run = async () => {
    body.set([c.hint.default({ text: "Checking…" })]);
    let rows;
    try { rows = (await api.readiness()).rows; } catch (err) { return body.set([c.banner.warn({ text: `Could not check: ${err.message}` }), check]); }
    const bad = rows.filter((r) => r.level !== "ok").length;
    body.set([check, bad ? c.banner.warn({ text: `${bad} thing${bad > 1 ? "s" : ""} to look at.` }) : c.banner.info({ text: "Nothing in the way." }),
      ...[...rows].sort((a, b) => ORDER[a.level] - ORDER[b.level]).map((r) => c.settingsRow.default({ label: r.text, control: c.chip[TONE[r.level]]({ label: MARK[r.level] }) }))]);
  };
  const check = c.button.sm({ label: "Check this machine", tone: "ghost", title: "Look for what a first run on a new GPU box trips over", onClick: run });
  body.set([c.hint.default({ text: "Run this on a new rental before the first generation: it checks ffmpeg, the GPU, model files, installed nodes and loaded modules." }), check]);
  return body;
};
