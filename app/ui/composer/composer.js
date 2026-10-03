// The kit's public surface.
//
// A module receives exactly this and nothing else: element factories, reached by
// name. There is no handle here to el(), to document, or to any internal -- which
// is what makes "a module never styles anything" a fact about the code rather
// than a rule people are asked to remember.

import "./elements/index.js";       // elements register themselves on import
import { composer } from "./internals/register.js";

export { composer };
export default composer;

// Re-exported for the shell and the catalogue, which legitimately need to know
// what registered. Modules never import this file directly.
export { entries, has, UnknownElement } from "./internals/register.js";
