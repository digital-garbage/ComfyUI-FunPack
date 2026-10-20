// The loading window: opens as soon as the page starts, names the module being loaded, closes when all are in.
// Built only from the kit. It cannot be dismissed, because the screen behind it is still filling in.
const name = (path) => {
  const parts = path.replace(/^.*\/features\//, "").replace(/\.js$/, "").split("/");
  if (parts[parts.length - 1] === "index") parts.pop();
  return parts.join("/") || path;
};

export function startLoading(c) {
  let win = null, text = null, bar = null;
  try {
    text = c.text.md({ center: true, text: "Starting FunPack. Please wait for UI to finish loading." });
    bar = c.progress.bar({ value: 0, max: 100, label: "Loading modules" });
    win = c.modal.generic({
      title: "Loading FunPack", size: "md", centered: true, closeOnOutside: false, closeOnEsc: false,
      body: c.region.stack({ gap: "lg", children: [text, bar] }),
    });
  } catch (err) { console.warn("[FunPack] loading window failed:", err.message); }      // no window is better than no app
  return {
    progress({ n, of, path }) {
      if (!win) return;
      text.setText(`Loading module ${n} out of ${of}: ${name(path)}. Please wait for UI to finish loading.`);
      bar.setValue(((n - 1) / of) * 100);
    },
    done() { if (win) win.close("done"); win = null; },
  };
}
