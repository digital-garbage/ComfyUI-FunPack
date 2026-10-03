// About FunPack: which build this is, and the machine ComfyUI runs on (which on a rental is not this browser's).
import { composer as c } from "../../composer/composer.js";

const facts = (rows) => rows.filter(([, v]) => v).map(([k, v]) => c.settingsRow.default({ label: k, control: c.text.sm({ text: String(v) }) }));

export const about = (api) => function mount() {
  const body = c.region.stack({ gap: "sm", label: "About" });
  body.set([c.hint.default({ text: "Looking…" })]);
  Promise.all([api.gitStatus().catch(() => ({})), api.system().catch(() => null)]).then(([git, sys]) => {
    const major = String(git.version || "").split(".")[0];
    const mem = (sys && sys.memory) || {}, cpu = (sys && sys.cpu) || {}, disk = (sys && sys.disk) || {}, torch = (sys && sys.torch) || {}, gpu = ((sys && sys.gpus) || [])[0];
    body.set([
      c.header.md({ text: `FunPack${major ? ` ${major}` : ""}` }),
      c.hint.default({ text: [git.codename && `“${git.codename}”`, "Cutting Room"].filter(Boolean).join(" · ") }),
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
