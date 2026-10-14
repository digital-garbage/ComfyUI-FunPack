// A project that used a module this machine does not have yet (a new rental) says so, with a way to Settings ▸ Modules.
import { composer as c } from "../../composer/composer.js";

export const RESTART_HINT = "Enabling this module requires installation of its dependencies, and will be performed on a next ComfyUI restart.";

export default {
  id: "addons",
  mount: "menubar.right",
  needs: ["api", "project"],
  setup({ app }) {
    let asked = null;                                       // the project id already told about: once per open
    const check = async () => {
      const p = app.project.project;
      const used = p && (p.editor_settings || {}).addons;
      if (!p || asked === p.id || !Array.isArray(used) || !used.length) return;
      asked = p.id;
      let all; try { all = await app.api.addons(); } catch { return; }
      const n = all.filter((a) => used.includes(a.id) && (a.state === "available" || a.state === "disabled")).length;
      if (!n) return;
      const link = c.button.sm({ label: "Modules", tone: "ghost", onClick: () => app.openSettings("modules") });
      c.toast.info({ text: `You have ${n} module(s) available for installation.`, action: link, duration: 12000 });
    };
    const off = app.on((what) => { if (what === "open") check(); });
    check();
    return off;
  },
};
