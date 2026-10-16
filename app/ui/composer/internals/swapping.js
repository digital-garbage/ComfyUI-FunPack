// Set while region.stack swaps its content. Chromium fires change/blur on the focused field as it is removed (still
// attached): a commit then would redraw the owner inside the swap, under the stale content the swap goes on to insert.
// A field skips committing meanwhile; region.stack carries its edit into the redrawn field instead.
export const swapping = { depth: 0 };
