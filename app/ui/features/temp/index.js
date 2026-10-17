import { mount } from "./view.js";

export default {
  id: "temp",
  mount: "settings",
  needs: ["dom"],
  setup: ({ host, app }) => host.add({
    id: "temp", group: "System", title: "Temp Files", subtitle: "Where a file went when it did not land in the bin.",
    keywords: "temp files output directory", icon: "▥", mount: (ctx) => mount({ ...ctx, dom: app.dom }),
  }),
};
