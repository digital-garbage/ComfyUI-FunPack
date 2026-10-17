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
