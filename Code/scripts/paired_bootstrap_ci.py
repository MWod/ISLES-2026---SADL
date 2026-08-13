"""Paired bootstrap 95% CIs and one-sided p-values for decision-layer eval.

Reads ``decision_layer_eval.json`` (baseline vs new per-case metrics on the
91-case ISLES V2 holdout) and reports:

* Overall paired mean Δ + 95% CI + one-sided p-value for each metric.
* Per-bucket paired mean Δ + 95% CI + p-value.

Δ convention:
* dice / lesion_f1 / count_f1 : higher-is-better, Δ = new - base, p = P(Δ ≤ 0).
* avd                         : lower-is-better,  Δ = new - base, p = P(Δ ≥ 0).

Bootstrap: BCa-adjusted CI (bias + acceleration) on paired resamples of case
indices. Default N=10000, RNG seed=42 for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = [
    ("dice", "higher"),
    ("lesion_f1", "higher"),
    ("count_f1", "higher"),
    ("avd", "lower"),
]


def bca_ci(diffs: np.ndarray, boot_diffs: np.ndarray, alpha: float) -> tuple[float, float]:
    """BCa-adjusted CI. Falls back to percentile CI if acceleration undefined."""
    theta_hat = diffs.mean()
    z0 = stats.norm.ppf((boot_diffs < theta_hat).mean())
    n = len(diffs)
    jack = np.array([np.delete(diffs, i).mean() for i in range(n)])
    jack_mean = jack.mean()
    num = ((jack_mean - jack) ** 3).sum()
    den = 6.0 * ((jack_mean - jack) ** 2).sum() ** 1.5
    a = num / den if den > 0 else 0.0
    zl = stats.norm.ppf(alpha / 2)
    zu = stats.norm.ppf(1 - alpha / 2)
    lo_q = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    hi_q = stats.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    return float(np.quantile(boot_diffs, lo_q)), float(np.quantile(boot_diffs, hi_q))


def paired_bootstrap(base: np.ndarray, new: np.ndarray, n_boot: int, seed: int, direction: str) -> dict:
    """Paired bootstrap Δ (new - base) with BCa CI and one-sided p-value."""
    rng = np.random.default_rng(seed)
    diffs = new - base
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_diffs = diffs[idx].mean(axis=1)
    lo, hi = bca_ci(diffs, boot_diffs, alpha=0.05)
    pval = float((boot_diffs <= 0).mean()) if direction == "higher" else float((boot_diffs >= 0).mean())
    return {
        "n": int(n),
        "base_mean": float(base.mean()),
        "new_mean": float(new.mean()),
        "delta_mean": float(diffs.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_one_sided": pval,
        "significant_at_0.05": pval < 0.05,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-json", type=Path, default=Path("Results/_diagnostics_pillar1_v2/decision_layer_eval.json")
    )
    ap.add_argument(
        "--out-json", type=Path, default=Path("Results/_diagnostics_pillar1_v2/paired_bootstrap_ci.json")
    )
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    payload = json.loads(args.eval_json.read_text())
    per_case = payload["per_case"]
    assert per_case, "empty per_case list"

    buckets = ["<0.5ml", "0.5-5ml", "5-50ml", ">=50ml"]
    results: dict[str, dict] = {"n_boot": args.n_boot, "seed": args.seed}

    for metric, direction in METRICS:
        base = np.array([c[f"base_{metric}"] for c in per_case], dtype=float)
        new = np.array([c[f"new_{metric}"] for c in per_case], dtype=float)
        results[metric] = {
            "direction": direction,
            "overall": paired_bootstrap(base, new, args.n_boot, args.seed, direction),
            "by_bucket": {},
        }
        for bucket in buckets:
            mask = np.array([c["bucket"] == bucket for c in per_case])
            if mask.sum() < 3:
                results[metric]["by_bucket"][bucket] = {"n": int(mask.sum()), "skipped": True}
                continue
            results[metric]["by_bucket"][bucket] = paired_bootstrap(
                base[mask], new[mask], args.n_boot, args.seed, direction
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))

    print(f"\nPaired bootstrap ({args.n_boot} resamples, seed={args.seed})")
    print(f"Source: {args.eval_json}")
    print(f"Wrote:  {args.out_json}\n")

    for metric, _direction in METRICS:
        ov = results[metric]["overall"]
        arrow = "↑" if results[metric]["direction"] == "higher" else "↓"
        star = "★" if ov["significant_at_0.05"] else " "
        print(f"=== {metric} ({arrow} better) ===")
        print(
            f"  overall (n={ov['n']:>3}) base={ov['base_mean']:.4f} → new={ov['new_mean']:.4f}"
            f"  Δ={ov['delta_mean']:+.4f}  95% CI [{ov['ci_lo']:+.4f}, {ov['ci_hi']:+.4f}]"
            f"  p={ov['p_one_sided']:.4f} {star}"
        )
        for bucket in buckets:
            b = results[metric]["by_bucket"][bucket]
            if b.get("skipped"):
                print(f"  {bucket:>7} (n={b['n']:>3}) skipped (n<3)")
                continue
            s = "★" if b["significant_at_0.05"] else " "
            print(
                f"  {bucket:>7} (n={b['n']:>3}) base={b['base_mean']:.4f} → new={b['new_mean']:.4f}"
                f"  Δ={b['delta_mean']:+.4f}  95% CI [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]"
                f"  p={b['p_one_sided']:.4f} {s}"
            )
        print()


if __name__ == "__main__":
    main()
