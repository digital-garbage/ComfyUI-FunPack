export function setup({ composer }) {
  const b = composer.button.md({ label: "x" });
  b.node.style.color = "#ff0000";
  b.node.classList.add("mine");
  b.node.innerHTML = "<b>hi</b>";
  const smuggled = '<div style="color: red">still styling</div>';
  const raw = document.createElement("div");
  raw.setAttribute("style", "margin: 12px");
  document.head.appendChild(raw);
  void smuggled;
  return new Function("return 1")();
}
