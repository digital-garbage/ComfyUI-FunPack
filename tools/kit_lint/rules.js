// What a module may not do, and what the kit may not do.
//
// These are the static half of rules 1 and 4. The runtime half is stronger --
// a module is never handed anything that could style or parse -- but a scanner
// catches the attempt in the file rather than at the moment it would have run,
// which is the difference between a failing test and a support ticket.

export const MODULE_RULES = [
  { name: "inline-style", why: "presentation belongs to the kit",
    test: /\.style\s*[.[]|setAttribute\s*\(\s*['"`]style/ },
  { name: "style-attribute", why: "presentation belongs to the kit",
    test: /\bstyle\s*=\s*['"`]/ },
  { name: "class-manipulation", why: "kit classes are the kit's; a module names elements, not classes",
    test: /\bclassList\b|setAttribute\s*\(\s*['"`]class/ },
  { name: "raw-element", why: "every element comes from composer.<group>.<variant>()",
    test: /document\s*\.\s*createElement|createElementNS/ },
  { name: "markup", why: "content is data; the kit renders it as text",
    test: /\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML|document\s*\.\s*write/ },
  { name: "stylesheet", why: "a module cannot ship styling, and cannot inject it either",
    test: /adoptedStyleSheets|document\s*\.\s*head|<\s*(link|style)\b/ },
  { name: "hardcoded-colour", why: "colour comes from the theme tokens",
    test: /#[0-9a-fA-F]{3,8}\b/ },
  { name: "hardcoded-size", why: "sizes come from the spacing and control scales",
    test: /\b\d+(\.\d+)?px\b/ },
  { name: "eval", why: "nothing in a module is compiled at runtime",
    test: /\beval\s*\(|new\s+Function\s*\(/ },
];

// The kit itself may style and may build elements -- that is its job -- but it
// must never interpret content.
export const KIT_RULES = [
  { name: "markup", why: "the kit renders content as text and never parses it",
    test: /\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML|document\s*\.\s*write/ },
  { name: "eval", why: "nothing in the kit is compiled at runtime",
    test: /\beval\s*\(|new\s+Function\s*\(/ },
];
