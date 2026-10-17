import { mount } from "./view.js";

export default {
  id: "temp",
  mount: "settings",
  needs: ["dom"],
  setup: ({ host, app }) => host.add({
    id: "temp", group: "System", title: "Saved & Temp Files", subtitle: "Your saved videos and pictures, and where a file went when it did not land in the bin.",
    keywords: "gallery saved temp files output directory videos pictures delete", icon: "▥", mount: (ctx) => mount({ ...ctx, dom: app.dom }),
  }),
};
