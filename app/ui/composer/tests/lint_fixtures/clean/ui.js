export function setup({ composer, values }) {
  const button = composer.button.md({
    label: "Re-key",
    onClick: () => values.set("enabled", true),
  });
  return button;
}
