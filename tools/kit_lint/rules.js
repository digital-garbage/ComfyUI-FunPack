// What the kit and the shell may not do.
//
// A DISCIPLINE CHECK, NOT A BOUNDARY: it catches the accidental in the file
// rather than at the moment it would have run. Modules need no rules here
// because they ship no JavaScript.

// The kit may style and build elements -- that is its job -- but it must never
// interpret content.
export const KIT_RULES = [
  { name: "markup", why: "the kit renders content as text and never parses it",
    test: /\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML|document\s*\.\s*write/ },
  { name: "eval", why: "nothing in the kit is compiled at runtime",
    test: /\beval\s*\(|new\s+Function\s*\(/ },
];

// A UI feature (app/ui/features) is app code. It builds with kit elements and reaches the page only
// through `app` (app.dom), so it names no element, class, colour or size of its own.
// localStorage is allowed (per-browser preferences).
export const FEATURE_RULES = [
  { name: "inline-style", why: "presentation belongs to the kit",
    test: /\.style\s*[.[]|\[\s*['"`]style['"`]\s*\]|setAttribute\s*\(\s*['"`]style/ },
  { name: "style-attribute", why: "presentation belongs to the kit",
    test: /\bstyle\s*=\s*['"`]/ },
  { name: "class-manipulation", why: "kit classes are the kit's; a feature names elements, not classes",
    test: /\bclassList\b|setAttribute\s*\(\s*['"`]class/ },
  { name: "class-name", why: "classes belong to the kit (cx-) and the frame (fp-); a feature names elements",
    test: /\bclassName\b|['"`.][^'"`\n]*\b(?:fp|cx)-[a-z]/ },
  { name: "document", why: "the page is reached through app.dom, never directly",
    test: /(?<![\w./])document(?:\.[A-Za-z_$]|\s*\[)|[(=,]\s*document\s*[,;)]|createElementNS|\[\s*['"`]createElement['"`]\s*\]/ },
  { name: "reaching-for-globals", why: "the page is reached through app.dom, never directly",
    test: /\bglobalThis\b|(?<![\w./])window(?:\.[A-Za-z_$]|\s*\[)|\bself\s*\[/ },
  { name: "markup", why: "content is data; the kit renders it as text",
    test: /\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML|\[\s*['"`](inner|outer)HTML['"`]\s*\]/ },
  { name: "stylesheet", why: "a feature cannot ship styling, and cannot inject it either",
    test: /adoptedStyleSheets|<\s*(link|style)\b/ },
  { name: "hardcoded-colour", why: "colour comes from the theme tokens",
    test: /#[0-9a-fA-F]{3,8}\b/ },
  { name: "hardcoded-size", why: "sizes come from the spacing and control scales",
    test: /\b\d+(\.\d+)?px\b/ },
  { name: "eval", why: "nothing in a feature is compiled at runtime",
    test: /\beval\s*[([]|new\s+Function\s*\(|\[\s*['"`](eval|Function)['"`]\s*\]/ },
];
