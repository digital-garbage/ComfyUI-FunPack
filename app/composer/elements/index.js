// Every element file, imported for its side effect: each one calls define() and
// so announces itself into the registry. This list is the only place element
// files are named, and it is the only enumeration in the kit -- ES modules give
// no way to discover siblings at runtime without a build step.
//
// Elements land here one at a time, each with its catalogue section and tests.
