// Settings sections that sit in v4's window: the real ones, and stand-ins (named, grouped, saying what is coming) for the rest.
import { composer as c } from "../../composer/composer.js";
import { about, readiness } from "./about.js";
import { appearance } from "./appearance.js";
import { models } from "./models.js";
import { engine } from "./engine.js";
import { modules } from "./modules.js";
import { system } from "./system.js";
import { packs } from "./packs.js";
import { taste } from "./taste.js";
import { editor } from "./editor.js";
import { shotMemory } from "./shotmemory.js";

export default {
  id: "settings-sections",
  mount: "settings",
  needs: ["api", "theme", "pipeline", "maintenance"],
  setup({ host, app }) {
    const add = (id, title, subtitle, icon, tone, mount, group = "", keywords = "") => host.add({ id, group, title, subtitle, icon, tone, mount, keywords: `${keywords} ${title}` });
    add("about", "About FunPack", "", "◎", "accent", about(app.api), "", "version commit branch cpu memory gpu disk python torch");
    add("readiness", "Ready to generate?", "Check this machine before a first run: ffmpeg, GPU, models, nodes, modules.", "✓", "accent", readiness(app.api), "", "check preflight rental gpu ffmpeg models missing nodes");
    add("appearance", "Appearance", "Light, dark, or follow the system.", "◐", "neutral", appearance(app.theme), "", "theme colour light dark auto");
    add("editor", "Editor", "How the editor behaves. The open project remembers these, so they follow it to another machine.", "☰", "neutral",
      editor(app), "", "upscale autocomplete shortcuts ideas preferences");
    add("engine", "Engine", "Settings every installed module has volunteered, grouped by what they're for.", "⚡", "warn",
      engine(app), "Generation", "engine module settings");
    add("models", "Models & Pipeline", "Loaders and nodes wired into the live pipeline.", "⬡", "accent",
      models(app), "Generation", "models loaders unet vae clip lora");
    add("modules", "Modules", "Switch a module off for this project; see and re-enable ones that failed.", "☷", "neutral",
      modules(app), "Generation", "modules enable disable quarantine");
    add("refinement", "Refinement & Taste", "Learned-taste state: refinement keys and the Absolute global-taste store.", "✦", "danger",
      taste(app), "Learning", "refinement taste keys rating");
    // only when that module is installed: a section for something absent would be a dead end
    app.pipeline.ensureLoaded().then(() => {
      if (app.pipeline.modulesById()["conditioning_shot_camera"]) add("shotmemory", "Shot camera memory", "What the camera has learned from your prompts and ratings.", "◎", "neutral", shotMemory(app), "Learning", "shot camera views moves words forget");
    });
    add("system", "Updates & ComfyUI", "Server connection, FunPack code updates, pipeline health.", "⟳", "good",
      system(app), "System", "update git branch restart rollback");
    add("customnodes", "Custom Nodes", "Install, update and remove ComfyUI node packs.", "⧉", "neutral",
      packs(app), "System", "custom nodes packs install");
  },
};
