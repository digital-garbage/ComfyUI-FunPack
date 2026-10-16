// Appearance: dark, light, or follow the system.
import { composer as c } from "../../composer/composer.js";

export const appearance = (theme) => function mount() {
  return c.settingsRow.default({ label: "Colour scheme", hint: "Light, dark, or follow the system.",
    control: c.segmented.sm({ label: "Colour scheme", value: theme.get(), onChange: (v) => theme.apply(v),
      options: [{ value: "dark", label: "Dark" }, { value: "light", label: "Light" }, { value: "auto", label: "Auto" }] }) });
};
