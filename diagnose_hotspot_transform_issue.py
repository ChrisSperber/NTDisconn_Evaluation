#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diagnostic tool for NT streamline weight vectors.

Usage:
    python diagnose_hotspot_transform_issue.py /path/to/HCP_NT/Percent

Optional:
    --t 1.5
    --alpha 3.0 # same value threshold as in the main pipeline
"""

import argparse
from pathlib import Path

import numpy as np


# -----------------------------
# robust sigmoid (same logic)
# -----------------------------
def robust_sigmoid_weights(x: np.ndarray, t: float, alpha: float = 3.0):
    x = np.asarray(x, dtype=np.float64)

    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad

    if scale <= 1e-12 or not np.isfinite(scale):
        # Degenerate case (matches main script logic)
        w = np.ones_like(x, dtype=np.float64)
        return w, med, mad, scale, True

    z = (x - med) / scale
    w = 1.0 / (1.0 + np.exp(-alpha * (z - t)))
    return w, med, mad, scale, False


# -----------------------------
# find weight files
# -----------------------------
def find_weight_files(root: Path):
    return sorted(root.rglob("*_weights_disc_Tractogram.txt"))


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Root folder of NT weight files")
    ap.add_argument("--t", type=float, default=1.5, help="Sigmoid threshold")
    ap.add_argument("--alpha", type=float, default=3.0)
    args = ap.parse_args()

    files = find_weight_files(args.root)

    if not files:
        print("No weight files found.")
        return

    print(f"Found {len(files)} NT weight files\n")

    header = (
        "NT".ljust(32)
        + "MAD".rjust(12)
        + "scale".rjust(12)
        + "mean_w".rjust(12)
        + "p>0.99".rjust(10)
        + "p<0.01".rjust(10)
        + "flags"
    )
    print(header)
    print("-" * len(header))

    for f in files:
        try:
            x = np.loadtxt(f)
        except Exception as e:
            print(f"{f.name:<32} ERROR reading: {e}")
            continue

        if x.ndim != 1:
            x = x.ravel()

        # compute transform
        w, med, mad, scale, mad_degenerate = robust_sigmoid_weights(
            x, t=args.t, alpha=args.alpha
        )

        mean_w = float(np.mean(w))
        p_high = float(np.mean(w > 0.99))
        p_low = float(np.mean(w < 0.01))
        denom = float(np.sum(w))

        flags = []

        # diagnostics
        if mad_degenerate:
            flags.append("MAD~0")

        if p_high < 1e-4:
            flags.append("no_hotspots")

        if p_high > 0.5:
            flags.append("too_binary_high")

        if mean_w < 1e-4:
            flags.append("denom_small")

        if not np.isfinite(denom) or denom <= 0:
            flags.append("bad_sum")

        nt_name = f.parent.name

        print(
            nt_name.ljust(32)
            + f"{mad:12.4g}"
            + f"{scale:12.4g}"
            + f"{mean_w:12.4g}"
            + f"{p_high:10.4g}"
            + f"{p_low:10.4g}"
            + ("  " + ",".join(flags) if flags else "")
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
