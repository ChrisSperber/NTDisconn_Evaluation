#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inspect how much "signal" remains in NT streamline weight maps after applying
the sparse-aware robust-z + sigmoid hotspot transform used in Create_NTDisconn.py.

It reports, per NT weight file:
  - how many values are non-zero in the raw map
  - median/MAD/IQR of the non-zero tail
  - after transform (for a given t, alpha):
      * mean(w), sum(w)
      * fraction w > 0.5, >0.9, >0.99 (overall + within non-zero tail)
      * an "effective number of streamlines" (ESS) based on weight concentration

Usage:
  python nt_hotspot_diagnostics.py /path/to/HCP_NT/Percent --t 3.0
  python nt_hotspot_diagnostics.py /path/to/HCP_NT/Percent --t 3.5 --alpha 5.0
"""

import argparse
from pathlib import Path

import numpy as np


def find_weight_files(root: Path):
    return sorted(root.rglob("*_weights_disc_Tractogram.txt"))


def sparse_robust_sigmoid(
    x: np.ndarray,
    t: float,
    alpha: float = 3.0,
    zero_eps: float = 0.0,
    eps: float = 1e-12,
):
    """
    Sparse-aware robust-z on non-zero tail + sigmoid, keeping zeros at 0.

    Returns:
      w        transformed weights, same shape as x
      stats    dict with tail/scale diagnostics
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values (NaN/Inf) in weights.")

    nz = x > float(zero_eps)
    n = x.size
    n_nz = int(nz.sum())

    if n_nz == 0:
        raise ValueError("All values are zero (or <= zero_eps).")

    x_nz = x[nz]
    med = float(np.median(x_nz))
    mad = float(np.median(np.abs(x_nz - med)))
    scale = float(1.4826 * mad)

    # If MAD collapses due to ties/discretization, fall back to IQR (on non-zero tail)
    if not np.isfinite(scale) or scale < eps:
        q25, q75 = np.percentile(x_nz, [25, 75])
        iqr = float(q75 - q25)
        scale = iqr
    else:
        iqr = float(np.percentile(x_nz, 75) - np.percentile(x_nz, 25))

    if not np.isfinite(scale) or scale < eps:
        raise ValueError(
            f"Non-zero tail dispersion ~0 (MAD={mad:.3g}, IQR={iqr:.3g}); cannot standardize."
        )

    z = (x_nz - med) / scale
    w_nz = 1.0 / (1.0 + np.exp(-float(alpha) * (z - float(t))))

    w = np.zeros_like(x, dtype=np.float64)
    w[nz] = w_nz

    stats = {
        "n": n,
        "n_nz": n_nz,
        "frac_nz": n_nz / float(n),
        "med_nz": med,
        "mad_nz": mad,
        "iqr_nz": iqr,
        "scale_used": scale,
    }
    return w, stats


def frac_gt(a: np.ndarray, thr: float, mask: np.ndarray | None = None) -> float:
    if mask is None:
        return float(np.mean(a > thr))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(a[mask] > thr))


def ess(weights: np.ndarray) -> float:
    """
    Effective number of streamlines with weight mass (a concentration metric).
    ESS = (sum w)^2 / sum(w^2)
    - equals N when all weights are equal
    - approaches 1 when one element dominates
    """
    w = np.asarray(weights, dtype=np.float64)
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s1 <= 0 or s2 <= 0:
        return 0.0
    return (s1 * s1) / s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root", type=Path, help="Root folder containing NT weight txt files"
    )
    ap.add_argument(
        "--t", type=float, required=True, help="Hotspot threshold t (robust-z units)"
    )
    ap.add_argument(
        "--alpha", type=float, default=3.0, help="Sigmoid steepness (default 3.0)"
    )
    ap.add_argument(
        "--zero_eps",
        type=float,
        default=0.0,
        help="Treat values <= zero_eps as zero (default 0.0)",
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Optional limit on number of files (0 = no limit)",
    )
    args = ap.parse_args()

    files = find_weight_files(args.root)
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        print("No '*_weights_disc_Tractogram.txt' files found under:", args.root)
        return

    print(f"Found {len(files)} weight files under: {args.root}")
    print(f"Using t={args.t}, alpha={args.alpha}, zero_eps={args.zero_eps}\n")

    header = (
        "NT".ljust(30)
        + "frac_nz".rjust(10)
        + "scale".rjust(10)
        + "mean(w)".rjust(10)
        + "p>0.5".rjust(8)
        + "p>0.9".rjust(8)
        + "p>0.99".rjust(9)
        + "p>0.9_nz".rjust(10)
        + "ESS%".rjust(8)
        + "flags"
    )
    print(header)
    print("-" * len(header))

    for f in files:
        nt_name = f.parent.name
        try:
            x = np.loadtxt(f)
        except Exception as e:
            print(nt_name.ljust(30) + f" ERROR read: {e}")
            continue

        x = np.asarray(x, dtype=np.float64).ravel()
        raw_nz = x > args.zero_eps

        try:
            w, st = sparse_robust_sigmoid(
                x, t=args.t, alpha=args.alpha, zero_eps=args.zero_eps
            )
        except Exception as e:
            print(nt_name.ljust(30) + f" ERROR transform: {e}")
            continue

        mean_w = float(np.mean(w))
        p05 = frac_gt(w, 0.5)
        p09 = frac_gt(w, 0.9)
        p099 = frac_gt(w, 0.99)
        p09_nz = frac_gt(w, 0.9, mask=raw_nz)

        ess_val = ess(w)
        ess_pct = 100.0 * ess_val / float(st["n"])

        flags = []
        # Very sparse hotspots: essentially nothing reaches ~1
        if p09_nz < 1e-3:  # <0.1% of nonzero tail
            flags.append("very_sparse")
        # Extremely binary (lots of high weights)
        if p09_nz > 0.2:
            flags.append("too_dense")
        # Weight mass extremely concentrated
        if ess_pct < 0.1:
            flags.append("mass_concentrated")

        print(
            nt_name.ljust(30)
            + f"{st['frac_nz']:10.3g}"
            + f"{st['scale_used']:10.3g}"
            + f"{mean_w:10.3g}"
            + f"{p05:8.3g}"
            + f"{p09:8.3g}"
            + f"{p099:9.3g}"
            + f"{p09_nz:10.3g}"
            + f"{ess_pct:8.3g}"
            + ("  " + ",".join(flags) if flags else "")
        )

    print("\nNotes:")
    print("  frac_nz   = fraction of raw weights > zero_eps")
    print(
        "  scale     = robust scale used on non-zero tail (MAD*1.4826 or IQR fallback)"
    )
    print("  p>0.9_nz  = fraction of non-zero tail weights with transformed w > 0.9")
    print(
        "  ESS%      = effective number of streamlines carrying weight mass, as % of N"
    )
    print("            (lower means mass concentrated into fewer streamlines)")
    print("\nDone.")


if __name__ == "__main__":
    main()
