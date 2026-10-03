// Module-supplied content is DATA, never markup.
//
// asText is the only way text reaches the DOM in this kit. It does not parse,
// sanitise or escape -- it refuses anything that is not a scalar, so "markup"
// is not a state the kit can be in. That is a stronger guarantee than escaping,
// which is a thing you can forget to do in one place.

const SCALAR = new Set(["string", "number", "boolean"]);

export function asText(value) {
  if (value == null) return document.createTextNode("");
  if (!SCALAR.has(typeof value)) {
    throw new TypeError(
      `Composer content must be a string, number or boolean; got ${describe(value)}. ` +
      "Elements take data, not nodes or markup."
    );
  }
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new TypeError(`Composer content must be a finite number; got ${value}.`);
  }
  return document.createTextNode(String(value));
}

/** Replace a node's content with `value`, rendered as text. */
export function setText(node, value) {
  while (node.firstChild) node.removeChild(node.firstChild);
  node.appendChild(asText(value));
  return node;
}

function describe(value) {
  if (typeof value === "object" && value !== null) {
    if (typeof value.nodeType === "number") return `a DOM node (<${value.nodeName?.toLowerCase()}>)`;
    if (Array.isArray(value)) return "an array";
    return "an object";
  }
  return `a ${typeof value}`;
}
