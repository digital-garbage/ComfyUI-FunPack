// Entry point. Increment 2: the kit's internals exist and the module graph
// loads; no elements are defined yet, so there is nothing to render.

import { entries } from "./composer/composer.js";

const count = [...entries()].length;
document.querySelector("#app").textContent =
  count === 0
    ? "FunPack v5 — Composer internals loaded, no elements defined yet"
    : `FunPack v5 — ${count} elements registered`;
