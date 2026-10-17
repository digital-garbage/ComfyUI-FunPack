import { composer } from "../../composer/composer.js";
import { open } from "./window.js";

export default {
  id: "log",
  mount: "menubar.right",
  needs: ["dom"],
  setup: ({ host, app }) => host.append(composer.button.sm({ label: "Log", onClick: () => open({ dom: app.dom }) }).node),
};
