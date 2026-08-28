# Bundled fonts

Self-hosted so the editor renders correctly offline, on air-gapped machines and
behind firewalls that block Google's CDN. v4 linked Google Fonts and rendered in
a fallback face for those users.

| Family | Role | Files | Licence |
|---|---|---|---|
| Figtree | body / UI | variable 300–900, latin + latin-ext | [OFL 1.1](OFL-Figtree.txt) — Copyright 2022 The Figtree Project Authors |
| Archivo | display | variable 400–900, latin + latin-ext | [OFL 1.1](OFL-Archivo.txt) — Copyright 2020 The Archivo Project Authors |
| IBM Plex Mono | mono | 400 and 500, latin + latin-ext | [OFL 1.1](OFL-IBMPlexMono.txt) — Copyright 2017 IBM Corp. |

Subsetted to latin and latin-ext as served by Google Fonts; cyrillic, greek and
vietnamese are dropped. 133 KB total.

The SIL Open Font License permits bundling and redistribution, including inside
a GPL project, provided the licence travels with the fonts — which is why the
three OFL texts sit beside them. The fonts are not covered by FunPack's GPL and
are not modified.

Faces are declared in `app/composer/tokens/fonts.css`.
