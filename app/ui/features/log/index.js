import { composer } from "../../composer/composer.js";
import { open } from "./window.js";

export default {
  id: "log",
  mount: "menubar.right",
  setup: ({ host }) => host.append(composer.button.sm({ label: "Log", onClick: () => open() }).node),
};
