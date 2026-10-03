// The Simple | Editor switch in the menubar, and Simple's own surface: the one-prompt bar under the monitor.
import { composer as c } from "../../composer/composer.js";

const switcher = {
  id: "mode-switch",
  mount: "menubar.mode",
  needs: ["mode"],
  setup({ host, app }) {
    const seg = c.segmented.sm({ label: "Mode", value: app.mode.now, onChange: (v) => app.mode.set(v),
      options: [{ value: "simple", label: "Simple" }, { value: "editor", label: "Editor" }] });
    host.append(seg.node);
    return app.mode.on((v) => seg.setValue && seg.setValue(v));
  },
};

// "Describe the shot you want…" over two buttons that slide Assets and Properties in. Only shown in Simple mode.
const bar = {
  id: "simple-bar",
  mount: "preview.simple",
  needs: ["mode", "project", "panel"],
  setup({ host, app }) {
    const p = app.project;
    const area = c.textarea.md({ label: "Prompt", rows: 3, placeholder: "Describe the shot you want…", value: "",
      onInput: (v) => { if (p.selected) p.setText(p.selected.id, v); } });
    const row = c.toolbar.default({ items: [], trailing: [c.button.sm({ label: "Media", tone: "ghost", onClick: () => app.panel("assets") }),
      c.button.sm({ label: "Advanced settings", tone: "ghost", onClick: () => app.panel("props") })] });
    host.append(area.node, row.node);
    const draw = () => {
      const simple = app.mode.now === "simple";
      area.node.hidden = row.node.hidden = !simple;
      if (!simple) app.closePanels();
      else if (!area.node.matches(":focus")) area.setValue(p.selected ? p.selected.text || "" : "");
    };
    draw();
    const off = [app.mode.on(draw), app.on(draw)];
    return () => off.forEach((f) => f());
  },
};

// A slid-over zone covers the button that opened it, so each carries its own close (Simple mode only).
const closer = (mount) => ({
  id: `panel-close:${mount}`, mount, needs: ["mode", "closePanels"],
  setup({ host, app }) {
    const b = c.button.sm({ label: "✕", tone: "ghost", title: "Close", onClick: () => app.closePanels() }).node;
    host.append(b);
    const draw = () => { b.hidden = app.mode.now !== "simple"; };
    draw();
    return app.mode.on(draw);
  },
});

export default [switcher, bar, closer("assets.actions"), closer("inspector.actions")];
