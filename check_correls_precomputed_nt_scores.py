from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations


def check_nt_weight_correlations(root_dir: Path) -> None:
    """
    Recursively search for .txt files in root_dir, load them as vectors,
    and compute pairwise Pearson correlations.

    Prints results to terminal.

    Assumptions:
        - one float per line
        - same number of lines per file (ideally 2M)
    """

    root_dir = Path(root_dir)

    txt_files = sorted(root_dir.rglob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {root_dir}")
        return

    print(f"Found {len(txt_files)} txt files.")
    print("Loading weight vectors...")

    data = {}
    lengths = {}

    # ---- load files ----
    for fp in txt_files:
        try:
            arr = np.loadtxt(fp, dtype=float)

            if arr.ndim != 1:
                raise ValueError("Not a 1D vector")

            if np.isnan(arr).any():
                print(f"[WARN] NaNs detected in {fp.name}")

            data[fp.name] = arr
            lengths[fp.name] = arr.size

        except Exception as e:
            print(f"[ERROR] Failed to load {fp}: {e}")

    if not data:
        print("No valid files loaded.")
        return

    # ---- length check ----
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        print("[ERROR] Not all files have the same length:")
        for k, v in lengths.items():
            print(f"  {k}: {v}")
        return

    n = next(iter(unique_lengths))
    print(f"All vectors length = {n}")

    # ---- compute correlations ----
    print("\nPairwise Pearson correlations:\n")

    results = []

    names = list(data.keys())

    for a, b in combinations(names, 2):
        r = np.corrcoef(data[a], data[b])[0, 1]
        results.append((a, b, r))
        print(f"{a}  vs  {b}  ->  r = {r:.4f}")

    # ---- optional summary ----
    if results:
        rs = [r for _, _, r in results]
        print("\nSummary:")
        print(f"  mean r = {np.mean(rs):.4f}")
        print(f"  min  r = {np.min(rs):.4f}")
        print(f"  max  r = {np.max(rs):.4f}")


check_nt_weight_correlations(Path(__file__).parent / "HCP_NT" / "Percent")
