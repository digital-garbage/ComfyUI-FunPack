// A feature's own nodes inside a row it shares with others: drawn again in the same spot, never moved to the end.
export function inPlace(host) {
  let mine = [];
  return (...next) => {
    if (mine.length === next.length && mine.length) mine.forEach((n, i) => n.replaceWith(next[i]));
    else { mine.forEach((n) => n.remove()); host.append(...next); }
    mine = next;
  };
}
