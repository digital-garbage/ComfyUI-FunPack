// Settings sections that sit in v4's window: the real ones, and stand-ins (named, grouped, saying what is coming) for the rest.
import { composer as c } from "../../composer/composer.js";
import { about } from "./about.js";
import { appearance } from "./appearance.js";
import { models } from "./models.js";
import { engine } from "./engine.js";
import { modules } from "./modules.js";
import { system } from "./system.js";
import { packs } from "./packs.js";

const later = (what) => function mount() {
  return c.emptyState.default({ icon: "◌", title: "Not built yet", hint: what });
};

export default {
  id: "settings-sections",
  mount: "settings",
  needs: ["api", "theme", "pipeline", "maintenance"],
  setup({ host, app }) {
    const add = (id, title, subtitle, icon, tone, mount, group = "", keywords = "") => host.add({ id, group, title, subtitle, icon, tone, mount, keywords: `${keywords} ${title}` });
    add("about", "About FunPack", "", "◎", "accent", about(app.api), "", "version commit branch cpu memory gpu disk python torch");
    add("appearance", "Appearance", "Light, dark, or follow the system.", "◐", "neutral", appearance(app.theme), "", "theme colour light dark auto");
    add("editor", "Editor", "How the editor behaves. The open project remembers these, so they follow it to another machine.", "☰", "neutral",
      later("Upscale model, prompt autocomplete, shortcut ideas, i2v options."), "", "upscale autocomplete shortcuts ideas preferences");
    add("engine", "Engine", "Settings every installed module has volunteered, grouped by what they're for.", "⚡", "warn",
      engine(app), "Generation", "engine module settings");
    add("models", "Models & Pipeline", "Loaders and nodes wired into the live pipeline.", "⬡", "accent",
      models(app), "Generation", "models loaders unet vae clip lora");
    add("modules", "Modules", "Switch a module off for this project; see and re-enable ones that failed.", "☷", "neutral",
      modules(app), "Generation", "modules enable disable quarantine");
    add("refinement", "Refinement & Taste", "Learned-taste state: refinement keys and the Absolute global-taste store.", "✦", "danger",
      later("Refinement keys: export, delete; the taste store."), "Learning", "refinement taste keys rating");
    add("system", "Updates & ComfyUI", "Server connection, FunPack code updates, pipeline health.", "⟳", "good",
      system(app), "System", "update git branch restart rollback");
    add("customnodes", "Custom Nodes", "Install, update and remove ComfyUI node packs.", "⧉", "neutral",
      packs(app), "System", "custom nodes packs install");
  },
};
