// Announcement, not enumeration -- the kit's own version of the rule the
// backend uses.
//
// `composer.button.large` exists only because elements/button.js called
// define("button", "large", ...). Nothing holds a list of elements, so adding
// one is a single file, and asking for one that was never defined throws rather
// than rendering something approximate. That throw is what makes a module with
// a bespoke element hide itself instead of shipping a broken panel.

export class UnknownElement extends Error {
  constructor(group, variant) {
    const known = variant === undefined
      ? `Known groups: ${[...groups()].join(", ") || "(none yet)"}.`
      : `Known variants of "${group}": ${[...variants(group)].join(", ") || "(none)"}.`;
    super(
      variant === undefined
        ? `composer.${group} is not a Composer element group. ${known}`
        : `composer.${group}.${variant} is not a Composer element. ${known}`
    );
    this.name = "UnknownElement";
    this.group = group;
    this.variant = variant;
  }
}

const registry = new Map(); // group -> Map(variant -> factory)

export function define(group, variant, factory) {
  if (typeof factory !== "function") {
    throw new TypeError(`define("${group}", "${variant}", ...) needs a factory function.`);
  }
  if (!registry.has(group)) registry.set(group, new Map());
  const g = registry.get(group);
  if (g.has(variant)) {
    // Two files claiming one name means one of them silently loses, and which
    // one depends on import order.
    throw new Error(`composer.${group}.${variant} is already defined.`);
  }
  g.set(variant, factory);
  return factory;
}

export const groups = () => registry.keys();
export const variants = (group) => (registry.get(group) || new Map()).keys();
export const has = (group, variant) => Boolean(registry.get(group)?.has(variant));
export const lookup = (group, variant) => registry.get(group)?.get(variant);

/** Every registered element, so a test can iterate what exists rather than a list. */
export function* entries() {
  for (const [group, gv] of registry) for (const [variant, factory] of gv) yield { group, variant, factory };
}

// Property access on a Proxy is also how JS probes an object (`then` when it is
// awaited, Symbol.toStringTag when it is printed). Those must answer undefined
// rather than throw, or a stray await turns into a confusing UnknownElement.
const probe = (key) => typeof key === "symbol" || key === "then" || key === "toJSON";

function groupProxy(group) {
  return new Proxy(Object.create(null), {
    get(_t, variant) {
      if (probe(variant)) return undefined;
      const factory = lookup(group, variant);
      if (!factory) throw new UnknownElement(group, variant);
      return factory;
    },
    has: (_t, variant) => has(group, variant),
    ownKeys: () => [...variants(group)],
    getOwnPropertyDescriptor: () => ({ enumerable: true, configurable: true }),
  });
}

export const composer = new Proxy(Object.create(null), {
  get(_t, group) {
    if (probe(group)) return undefined;
    if (!registry.has(group)) throw new UnknownElement(group);
    return groupProxy(group);
  },
  has: (_t, group) => registry.has(group),
  ownKeys: () => [...groups()],
  getOwnPropertyDescriptor: () => ({ enumerable: true, configurable: true }),
  set() {
    throw new TypeError("composer is read-only; elements register themselves with define().");
  },
});

// Tests only.
export function _clearRegistry() {
  registry.clear();
}
