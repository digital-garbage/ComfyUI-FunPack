// Things the app can be asked to do, by name: whoever builds a control offers it here, whoever shows a list (the wheel) reads what is there.
// A feature that failed to load offers nothing and is simply not among them.
export function offer(app, action) {
  const list = app.actions || (app.actions = []);
  list.push(action);
  return () => { const i = list.indexOf(action); if (i >= 0) list.splice(i, 1); };
}
