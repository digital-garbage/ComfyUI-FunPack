// Updates & ComfyUI: update, switch branch, restart, roll back.
import { composer as c } from "../../composer/composer.js";

export const system = (app) => function mount() {
  const m = app.maintenance;
  const rows = [["Update", "update", "update"], ["Switch branch", "branch", "switchBranch"], ["Restart ComfyUI", null, "restart"], ["Rollback update", "rollback", "rollback"]]
    .map(([label, hint, act]) => {
      const button = c.button.sm({ label, tone: act === "restart" ? "ghost" : "primary", disabled: Boolean(hint), onClick: () => m[act]() });
      return { label, hint, button, node: c.settingsRow.default({ label, hint: hint ? "Checking…" : "Reload the server without updating", control: button }) };
    });
  m.hints().then((h) => rows.forEach((r) => {
    if (!r.hint) return;
    r.button.setDisabled(!h.ok || (r.hint === "rollback" && !h.rollbackOk));
    r.node.node.querySelector(".cx-hint").textContent = h[r.hint];
  }));
  return c.region.stack({ gap: "sm", children: rows.map((r) => r.node) });
};
