"""Print the four-group comparison for a Taste key's block-influence recordings.

    python tools/taste_groups.py <taste key>

Reads the rows the recording wrote (liked, bad image, bad composition, both bad) and prints how
the group directions line up, next to what shuffled labels give by chance. Reads only; changes nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(key: str) -> None:
    from modules.sampling.block_influence import measure
    from modules.system.taste import store

    rows = store.load(key, measure.KIND)["rows"]
    out = measure.groups(rows)
    print(f"key {key!r}: {out['used']} clips rated")
    print("counts:", out.get("counts", {}))
    if out.get("note"):
        print(out["note"])
    for pair, cos in out.get("cos", {}).items():
        c = out["chance"][pair]
        print(f"{pair}: {cos:+.3f}   (chance: mean {c['mean']:+.3f}, 95th {c['p95']:+.3f})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
