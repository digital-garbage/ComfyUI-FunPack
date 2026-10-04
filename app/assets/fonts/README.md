# Bundled fonts

Self-hosted so the editor renders correctly offline, on air-gapped machines and
behind firewalls that block Google's CDN. v4 linked Google Fonts and rendered in
a fallback face for those users.

| Family | Role | Files | Licence |
|---|---|---|---|
| Inter | body / UI (the app in `app/ui`) | variable 400–700, latin + latin-ext | [OFL 1.1](OFL-Inter.txt) — Copyright 2016 The Inter Project Authors |
| Inter Tight | display (the app in `app/ui`) | variable 600–800, latin + latin-ext | [OFL 1.1](OFL-InterTight.txt) — Copyright 2016 The Inter Project Authors |
| IBM Plex Mono | mono | 400 and 500, latin + latin-ext | [OFL 1.1](OFL-IBMPlexMono.txt) — Copyright 2017 IBM Corp. |

Subsetted to latin and latin-ext as served by Google Fonts; cyrillic, greek and
vietnamese are dropped. 133 KB total.

The SIL Open Font License permits bundling and redistribution, including inside
a GPL project, provided the licence travels with the fonts — which is why the
three OFL texts sit beside them. The fonts are not covered by FunPack's GPL and
are not modified.

Faces are declared in `app/ui/composer/tokens/fonts.css`.
