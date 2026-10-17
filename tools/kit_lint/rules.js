// What a module may not do, and what the kit may not do.
//
// A DISCIPLINE CHECK, NOT A BOUNDARY. Say it plainly, because the difference
// matters: a module's ui.js is an ES module in the page realm, so `document` is
// a global it can reach and no regex can stop it. `globalThis["docu"+"ment"]`
// defeats every rule below and always will; only real isolation -- a worker,
// another realm -- would be a boundary.
//
// What this does buy: the accidental and the casual are caught in the file
// rather than at the moment they would have run, which is the difference
// between a failing test and a support ticket. The bracket-access spellings are
// here because they were the first thing tried and they went straight through.
//
// The runtime half is the real work: the kit hands a module nothing that
// touches the DOM, so following the contract requires no restraint.

export const MODULE_RULES = [
  { name: "inline-style", why: "presentation belongs to the kit",
    test: /\.style\s*[.[]|\[\s*['"`]style['"`]\s*\]|setAttribute\s*\(\s*['"`]style/ },
  { name: "style-attribute", why: "presentation belongs to the kit",
    test: /\bstyle\s*=\s*['"`]/ },
  { name: "class-manipulation", why: "kit classes are the kit's; a module names elements, not classes",
    test: /\bclassList\b|setAttribute\s*\(\s*['"`]class/ },
  { name: "raw-element", why: "every element comes from the kit, not from a module",
    test: /document\s*[.[]|createElementNS|\[\s*['"`]createElement['"`]\s*\]/ },
  { name: "reaching-for-globals", why: "a module is handed what it may use; the rest is not for it",
    test: /\bglobalThis\b|\bwindow\s*[.[]|\bself\s*\[/ },
  { name: "markup", why: "content is data; the kit renders it as text",
    test: /\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML|\[\s*['"`](inner|outer)HTML['"`]\s*\]/ },
  { name: "stylesheet", why: "a module cannot ship styling, and cannot inject it either",
    test: /adoptedStyleSheets|document\s*\.\s*head|<\s*(link|style)\b/ },
  { name: "hardcoded-colour", why: "colour comes from the theme tokens",
    test: /#[0-9a-fA-F]{3,8}\b/ },
  { name: "hardcoded-size", why: "sizes come from the spacing and control scales",
    test: /\b\d+(\.\d+)?px\b/ },
  { name: "eval", why: "nothing in a module is compiled at runtime",
    test: /\beval\s*[([]|new\s+Function\s*\(|\[\s*['"`](eval|Function)['"`]\s*\]/ },
];

// The kit itself may style and may build elements -- that is its job -- but it
// must never interpret content.
export const KIT_RULES = [
  { name: "markup", why: "the kit renders content as text and never parses it",
    test: /\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML|document\s*\.\s*write/ },
  { name: "eval", why: "nothing in the kit is compiled at runtime",
    test: /\beval\s*\(|new\s+Function\s*\(/ },
];

// A UI feature (app/ui/features) is app code, not a module, but it follows the
// same discipline: it builds with kit elements and reaches the page only through
// `app`. What it may not do is the module list, plus the frame's and the kit's
// private class names. localStorage is allowed (per-browser preferences).
export const FEATURE_RULES = [
  ...MODULE_RULES.filter((r) => !["reaching-for-globals", "raw-element"].includes(r.name)),
  { name: "document", why: "the page is reached through app.dom, never directly",
    test: /(?<![\w./])document(?:\.[A-Za-z_$]|\s*\[)|[(=,]\s*document\s*[,;)]|createElementNS|\[\s*['"`]createElement['"`]\s*\]/ },
  { name: "reaching-for-globals", why: "the page is reached through app.dom, never directly",
    test: /\bglobalThis\b|(?<![\w./])window(?:\.[A-Za-z_$]|\s*\[)|\bself\s*\[/ },
  { name: "class-name", why: "classes belong to the kit (cx-) and the frame (fp-); a feature names elements",
    test: /\bclassName\b|['"`.][^'"`\n]*\b(?:fp|cx)-[a-z]/ },
];
