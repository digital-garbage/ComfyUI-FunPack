// Every element file, imported for its side effect: each one calls define() and
// so announces itself into the registry. This list is the only place element
// files are named, and it is the only enumeration in the kit -- ES modules give
// no way to discover siblings at runtime without a build step.
//
// Elements land here one at a time, each with its catalogue section and tests.

import "./text.js";
import "./button.js";
import "./input.js";
import "./choice.js";
import "./slider.js";
import "./layout.js";
import "./status.js";
import "./modal.js";
import "./popover.js";
import "./floating.js";
import "./gallery.js";
import "./wheel.js";
