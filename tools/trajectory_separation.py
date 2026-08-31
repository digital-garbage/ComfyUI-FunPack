#!/usr/bin/env python3
"""Read the trajectory probe's logs and answer one question: do ratings separate EARLY?

    python tools/trajectory_separation.py

Every rating-driven mechanism in the Scene Chain sampler is gated to the last half of the
schedule. The proposal that motivates this tool is to bank the model's own x0_hat per
schedule position and learn a value per position, so a "Missing action" rating can act in
the half where motion is actually decided. That is only worth building if the early half
carries signal at all.

Read the EARLY buckets (0, and 1 on a 4-bucket run). If their separation does not beat
label shuffling, an early-window value function would be training on noise and the idea is
dead as designed. The late buckets are the control: they are where the existing output
value function already works, so a run of the probe that shows nothing anywhere means the
measurement is broken, not that the model is.

See trajectory_probe.py for what is recorded and why.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trajectory_probe as probe  # noqa: E402


def _fmt(value, width=8, places=4):
    return "—".rjust(width) if value is None else f"{value:>{width}.{places}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key", help="Only this refinement key (default: pool every key)")
    parser.add_argument("--trials", type=int, default=2000, help="Label shuffles (default 2000)")
    parser.add_argument("--good-above", type=float, default=0.0,
                        help="Reward above this counts as a good run (default 0.0)")
    parser.add_argument("--bad-below", type=float, default=0.0,
                        help="Reward below this counts as a bad run (default 0.0)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Verdict threshold (default 0.05)")
    args = parser.parse_args()

    if args.key:
        logs = [(args.key, probe.load_log(args.key))]
    else:
        logs = probe.load_all_logs()
    rows = [row for _key, key_rows in logs for row in key_rows]

    if not rows:
        print("No recorded runs. Set FUNPACK_TRAJECTORY_PROBE=1, set a refinement key, "
              "generate, and rate — each rated generation adds one run.")
        return 1

    keys = [k for k, r in logs if r]
    print(f"{len(rows)} rated run(s) across {len(keys)} key(s): {', '.join(keys)}")

    mix = probe.schedule_mix(rows)
    if len(mix) > 1:
        print("\nWARNING — runs come from more than one schedule. A bucket covers a different "
              "sigma range on each, so pooling them compares different windows:")
        for (steps, sv, sa), count in sorted(mix.items(), key=lambda kv: str(kv[0])):
            print(f"  steps={steps} shift_video={sv} shift_audio={sa}: {count} run(s)")
        print("  Re-run with --key, or measure one schedule at a time.")

    unbound = sum(int(r.get("unbound_steps") or 0) for r in rows)
    if unbound:
        print(f"\nWARNING — {unbound} sampled step(s) across these runs were on a schedule "
              "the probe was never told about, and were refused rather than logged. Those "
              "runs are measured from fewer steps than they took.")
    multipass = [r for r in rows if len(r.get("pass_steps") or {}) > 1]
    if multipass:
        print(f"\n{len(multipass)} run(s) sampled more than one pass. Only the first pass is "
              "analysed: a second pass runs its own schedule from its own starting sigma, so "
              "its buckets cover different windows and are a separate measurement.")

    results = probe.analyse(rows, trials=args.trials,
                            good_above=args.good_above, bad_below=args.bad_below)
    if not results:
        print("Nothing to analyse — no run recorded any bucket.")
        return 1

    # One test per bucket, so the headline verdict uses a Bonferroni-corrected threshold.
    # At a raw 0.05 across four buckets, roughly one run in five would green-light building
    # the thing on a fluke — too loose for a decision this drives.
    threshold = args.alpha / max(1, len(results))
    print(f"\n{'bucket':>6} {'good':>5} {'bad':>4} {'separation':>11} {'p':>8} "
          f"{'noise floor':>12}  verdict (p <= {threshold:.4f}, "
          f"{args.alpha} corrected for {len(results)} buckets)")
    usable = False
    for entry in results:
        pooled = entry.get("pooled")
        label = "early (unreachable today)" if entry["bucket"] < len(results) / 2 else "late (already steered)"
        if pooled is None:
            print(f"{entry['bucket']:>6} {entry['n_good']:>5} {entry['n_bad']:>4} "
                  f"{'—':>11} {'—':>8} {'—':>12}  not enough runs — {label}")
            continue
        significant = pooled["p_value"] <= threshold
        usable = usable or (significant and entry["bucket"] < len(results) / 2)
        print(f"{entry['bucket']:>6} {entry['n_good']:>5} {entry['n_bad']:>4} "
              f"{_fmt(pooled['separation'], 11)} {_fmt(pooled['p_value'], 8, 4)} "
              f"{_fmt(pooled['noise_floor'], 12)}  "
              f"{'SEPARATES' if significant else 'no signal'} — {label}")
        within = entry.get("within_prompt")
        if within is not None:
            print(f"{'':>6} {'':>5} {'':>4} {_fmt(within['separation'], 11)} "
                  f"{_fmt(within['p_value'], 8, 4)} {_fmt(within['noise_floor'], 12)}  "
                  "  ^ same-prompt pairs only")

    print()
    if usable:
        print("VERDICT: an early bucket separates by more than label shuffling produces. "
              "A per-bucket value function has something to learn there — the early-window "
              "extension is worth building.")
    else:
        print("VERDICT: no early bucket beats its own label shuffles. As designed, an "
              "early-window value function would train on noise. Check the late buckets "
              "first: if they show nothing either, the measurement is at fault (too few "
              "runs, or every run rated the same way), not the model.")
    print("\nBoth columns matter: 'separation' can look large on a handful of runs, which "
          "is what p is for. Within-prompt rows, when present, are the honest ones — they "
          "rule out 'these were simply different prompts'. A permutation test cannot report "
          "a p below 1/(trials+1), and with few runs per prompt the stratified null has "
          "little resolution, so read a lone borderline row as 'measure more', not as a "
          "result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
