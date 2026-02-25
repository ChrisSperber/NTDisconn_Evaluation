#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import os
import shutil

import ants
import nibabel as nib
import numpy as np
import pandas as pd
import requests
from dipy.tracking._utils import _mapping_to_voxel, _to_voxel_coordinates
from tqdm import tqdm


def buildArgsParser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument("ID", help="Subject ID")
    p.add_argument("in_lesion", help="Input individual Lesionmask in MNI152 (1mm iso)")
    p.add_argument("output_dir", help="Specify output directory")
    p.add_argument(
        "--discStreamlines",
        default="y",
        help="Create disconnected streamline output? [y|n]",
    )
    p.add_argument(
        "--NTmaps", default="Percent", help="Which NT maps to use? [Z|Percent]"
    )
    p.add_argument(
        "--filter",
        default="n",
        help="Filter Streamlines - enter percentile [y|n]",
    )

    # NEW: optional hotspot-emphasizing transform of per-streamline NT weights (gtmap)
    #
    # If provided, this value is used as the robust-z threshold "t" in:
    #   z = (x - median(x)) / (1.4826 * MAD(x))
    #   w = sigmoid(alpha * (z - t))
    #
    # Intuition:
    #   - larger t -> only strong "hotspot" weights remain influential
    #   - smaller t -> more weights contribute (closer to original behavior)
    #
    # Suggested starting points:
    #   t=1.0  mild hotspot emphasis
    #   t=1.5  moderate
    #   t=2.0  strong
    #
    # IMPORTANT:
    #   This transform is only supported in --NTmaps Percent mode (proportion-style output).
    #   If --NTmaps Z is used together with this option, the script will fail.
    p.add_argument(
        "--nt_hotspot_t",
        type=float,
        default=None,
        help=(
            "Optional robust-sigmoid hotspot transform for gtmap weights. "
            "Provide threshold t (float) in robust-z units. "
            "Examples: 1.0 (mild), 1.5 (moderate), 2.0 (strong). "
            "Only valid with --NTmaps Percent; will error for --NTmaps Z."
        ),
    )

    # NEW: strict validation for streamline->voxel mapping
    p.add_argument(
        "--strict_voxel_indexing",
        default="n",
        help="Fail if any streamline point maps outside lesion/reference grid [y|n]",
    )
    p.add_argument(
        "--strict_nan_inf",
        default="n",
        help="Fail if NaN/Inf appear in warped lesion or NT weights [y|n]",
    )
    p.add_argument(
        "--max_oob_fraction",
        type=float,
        default=0.0,
        help=(
            "Allowed fraction of streamline points out of bounds before failing "
            "(only used when --strict_voxel_indexing y). Default 0.0"
        ),
    )

    return p


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _check_no_nan_inf(arr: np.ndarray, name: str) -> None:
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        _fail(f"[{name}] contains non-finite values: NaN={n_nan}, Inf={n_inf}")


def _transform_gtmap_robust_sigmoid(
    gtmap: np.ndarray,
    t: float,
    *,
    name: str = "gtmap",
    alpha: float = 3.0,
    eps: float = 1e-12,
    zero_eps: float = 0.0,  # treat <= zero_eps as "zero"
) -> np.ndarray:
    """
    Robust hotspot emphasis for sparse NT weights (many zeros + tail).

    Key idea:
      - If many weights are exactly 0, MAD around the global median can be 0.
      - In that case, standardize using the *non-zero* subset only.
      - Keep zeros exactly zero after transform.

    Returns weights in [0,1], suitable for proportion scoring:
        sum(disconnected*w) / sum(w)
    """
    x = np.asarray(gtmap, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{name}: gtmap must be 1D, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: gtmap contains NaN/Inf")

    # Identify "non-zero" tail
    nz = x > float(zero_eps)
    n = x.size
    n_nz = int(nz.sum())

    # If essentially all-zero, there's no usable weighting signal
    if n_nz == 0:
        raise RuntimeError(
            f"{name}: all weights are <= {zero_eps}; cannot apply hotspot transform."
        )

    # Compute robust center/scale on non-zero values
    x_nz = x[nz]
    med = np.median(x_nz)
    mad = np.median(np.abs(x_nz - med))
    scale = 1.4826 * mad

    # Fallback: use IQR of non-zero values if MAD is tiny (ties/discretization)
    if not np.isfinite(scale) or scale < eps:
        q25, q75 = np.percentile(x_nz, [25, 75])
        scale = float(q75 - q25)

    if not np.isfinite(scale) or scale < eps:
        raise RuntimeError(
            f"{name}: cannot apply hotspot transform; dispersion of non-zero tail is ~0 "
            f"(n={n}, n_nonzero={n_nz}, MAD={mad:.3g}, scale={scale:.3g})."
        )

    # Robust z for non-zero part only
    z = (x_nz - med) / scale

    # Sigmoid -> (0,1)
    w_nz = 1.0 / (1.0 + np.exp(-float(alpha) * (z - float(t))))

    # Reconstruct full vector; keep zeros exactly 0
    w = np.zeros_like(x, dtype=np.float64)
    w[nz] = w_nz

    # Safety: ensure denominator isn't degenerate
    if float(np.sum(w)) <= 0.0:
        raise RuntimeError(
            f"{name}: transformed weights have zero sum; check parameters."
        )
    return w


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    reference = "HCPA422-T1w-500um-norm.nii.gz"
    out_NT_disc = os.path.join(
        args.output_dir, args.ID + "_NT_Diconnect_" + args.NTmaps + ".csv"
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    strict_voxel = args.strict_voxel_indexing.lower() == "y"
    strict_nan_inf = args.strict_nan_inf.lower() == "y"

    # Enforce: hotspot transform only supported in Percent mode
    if args.nt_hotspot_t is not None and args.NTmaps != "Percent":
        _fail(
            "--nt_hotspot_t was provided, but --NTmaps is not 'Percent'. "
            "The hotspot transform is only implemented for proportion-style output (--NTmaps Percent). "
            "Please switch to --NTmaps Percent or omit --nt_hotspot_t."
        )

    def define_streamlines(streamlines, lesion, reference_img):
        """
        Return a 0/1 array indicating whether each streamline intersects the lesion.

        Safety checks (when enabled):
          - fail on negative indices
          - fail on indices >= shape
          - optionally allow a tiny fraction of OOB points via --max_oob_fraction
        """
        les = lesion.get_fdata()
        if strict_nan_inf:
            _check_no_nan_inf(les, "lesion_data")

        affine = reference_img.affine
        lin_T, offset = _mapping_to_voxel(affine)

        n_streamlines_expected = 2_000_000
        if len(streamlines) < n_streamlines_expected:
            _fail(
                f"Tractogram has only {len(streamlines)} streamlines, "
                f"but code expects {n_streamlines_expected}. "
                "Fix by iterating over range(len(streamlines)) or use the correct tractogram."
            )

        metric = np.zeros(n_streamlines_expected, dtype=np.float32)

        nx, ny, nz = les.shape[:3]

        total_points = 0
        oob_points = 0
        neg_points = 0
        high_points = 0

        for s in tqdm(range(n_streamlines_expected), desc="Evaluate streamlines"):
            streamline = streamlines[s]

            ijk = _to_voxel_coordinates(streamline[:], lin_T, offset)
            if ijk.ndim != 2 or ijk.shape[1] != 3:
                _fail(
                    f"Unexpected voxel coordinate shape for streamline {s}: {ijk.shape}"
                )

            x = ijk[:, 0]
            y = ijk[:, 1]
            z = ijk[:, 2]

            n_pts = int(len(x))
            total_points += n_pts

            neg_mask = (x < 0) | (y < 0) | (z < 0)
            high_mask = (x >= nx) | (y >= ny) | (z >= nz)
            oob_mask = neg_mask | high_mask

            n_neg = int(neg_mask.sum())
            n_high = int(high_mask.sum())
            n_oob = int(oob_mask.sum())

            neg_points += n_neg
            high_points += n_high
            oob_points += n_oob

            if strict_voxel and n_oob > 0:
                frac = n_oob / float(n_pts) if n_pts > 0 else 0.0
                if frac > args.max_oob_fraction:
                    bad_idx = np.where(oob_mask)[0][:5]
                    examples = [(int(x[i]), int(y[i]), int(z[i])) for i in bad_idx]
                    _fail(
                        "Streamline->voxel mapping produced out-of-bounds indices.\n"
                        f"  streamline_index: {s}\n"
                        f"  lesion_shape: {(nx, ny, nz)}\n"
                        f"  oob_points: {n_oob}/{n_pts} (frac={frac:.6f})\n"
                        f"  neg_points: {n_neg}, high_points: {n_high}\n"
                        f"  example_bad_ijk: {examples}\n"
                        "This can happen due to space mismatch between tractogram and reference affine, "
                        "or due to streamline points outside the image FOV."
                    )

            if n_oob > 0:
                keep = ~oob_mask
                if not np.any(keep):
                    continue
                xk = x[keep]
                yk = y[keep]
                zk = z[keep]
            else:
                xk, yk, zk = x, y, z

            if np.sum(les[xk, yk, zk]) > 0:
                metric[s] = 1.0

        if strict_voxel and total_points > 0:
            frac_oob = oob_points / float(total_points)
            print(
                f"[VOXEL QC] total_points={total_points} "
                f"oob_points={oob_points} (frac={frac_oob:.6e}) "
                f"neg_points={neg_points} high_points={high_points}"
            )

        return metric

    # ----- tractogram download -----
    tck_file = "HCP422_2_million.tck"
    if os.path.isfile(tck_file):
        print("Tactogram exists")
    else:
        print("Downloading Tractogram...........")
        osf_url = "https://osf.io/download/nduwc/"
        response = requests.get(osf_url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(tck_file, "wb") as file, tqdm(
            desc=tck_file, total=total_size, unit="B", unit_scale=True
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))
        print("Download complete!")

    # ----- registration MNI -> HCPA -----
    if os.path.isfile("MNI_to_HCPA_Warp.nii.gz") and os.path.isfile("MNI_to_HCPA.mat"):
        print("Coregistration MNI to HCPA done")
    else:
        print("Coregistration MNI to HCPA .....")
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
        mi = ants.image_read("MNI152_T1_1mm.nii.gz")
        fi = ants.image_read(reference)
        tx = ants.registration(
            fixed=fi,
            moving=mi,
            type_of_transform="SyN",
            random_seed=1,
            singleprecision=False,
        )
        forwardtrans = tx["fwdtransforms"]
        shutil.copyfile(forwardtrans[1], "MNI_to_HCPA.mat")
        shutil.copyfile(forwardtrans[0], "MNI_to_HCPA_Warp.nii.gz")
        print("Coregistration done!")

    out_weights_tractogram_disc = os.path.join(
        args.output_dir, args.ID + "_Disc_Streamlines.txt"
    )

    if os.path.isfile(out_weights_tractogram_disc):
        print("disc sl already calculated")
        weights_tractogram = np.loadtxt(out_weights_tractogram_disc).astype(np.float32)
    else:
        print("Loading streamlines ##########################################")
        tractogram = nib.streamlines.load(tck_file)
        streamlines = tractogram.streamlines

        # ----- warp lesion (MNI) -> HCPA -----
        standard = reference
        listtransf = ["MNI_to_HCPA_Warp.nii.gz", "MNI_to_HCPA.mat"]
        fi = ants.image_read(standard)
        movmap = ants.image_read(args.in_lesion)
        mywarpedimage = ants.apply_transforms(
            fixed=fi,
            moving=movmap,
            transformlist=listtransf,
            interpolator="multiLabel",
        )

        output = mywarpedimage.numpy()
        if strict_nan_inf:
            _check_no_nan_inf(output, "warped_lesion_numpy")

        ref = nib.load(standard)
        lesion = nib.Nifti1Image(output, ref.affine, ref.header)

        tmp_les_path = os.path.join(
            args.output_dir, f"{args.ID}_tmp_les_in_HCPA.nii.gz"
        )
        nib.save(lesion, tmp_les_path)

        weights_tractogram = define_streamlines(
            streamlines, lesion, nib.load(reference)
        )
        if strict_nan_inf:
            _check_no_nan_inf(weights_tractogram, "weights_tractogram")

        if args.discStreamlines == "y":
            np.savetxt(out_weights_tractogram_disc, weights_tractogram)

    if os.path.isfile(out_NT_disc):
        print("NT Disc already calculated")
        return

    d = {}
    d["ID"] = args.ID
    d["Disc_SL"] = float(np.sum(weights_tractogram))
    print("Evaluate NT systems......................")

    for neurotrans in [
        "5HT1a_way_hc36_savli",
        "5HT1b_p943_hc65_gallezot",
        "5HT2a_cimbi_hc29_beliveau",
        "5HT4_sb20_hc59_beliveau",
        "5HT6_gsk_hc30_radhakrishnan",
        "5HTT_dasb_hc100_beliveau",
        "D1_SCH23390_hc13_kaller",
        "D2_flb457_hc37_smith",
        "DAT_fpcit_hc174_dukart_spect",
        "A4B2_flubatine_hc30_hillmer",
        "VAChT_feobv_hc18_aghourian_sum",
        "mGluR5_abp_hc22_rosaneto",
        "GABAa-bz_flumazenil_hc16_norgaard",
        "NAT_MRB_hc77_ding",
        "H3_cban_hc8_gallezot",
        "M1_lsn_hc24_naganawa",
        "CB1_omar_hc77_normandin",
        "NMDA_ge179_hc29_galovic",
        "MU_carfentanil_hc204_kantonen",
    ]:
        print(neurotrans)
        in_neurotrans_weights = os.path.join(
            "HCP_NT",
            args.NTmaps,
            neurotrans,
            "GT_" + neurotrans + "_weights_disc_Tractogram.txt",
        )

        gtmap = np.loadtxt(in_neurotrans_weights).astype(np.float32)

        if strict_nan_inf:
            _check_no_nan_inf(gtmap, f"gtmap[{neurotrans}]")

        # filter by percentile (fixed 75th percentile if enabled)
        if args.filter != "n":
            cutoff = np.percentile(gtmap, 75)
            gtmap[gtmap < cutoff] = 0

        if gtmap.shape[0] != weights_tractogram.shape[0]:
            _fail(
                f"Length mismatch for {neurotrans}: "
                f"gtmap has {gtmap.shape[0]} entries but weights_tractogram has {weights_tractogram.shape[0]}."
            )

        # Optional: robust hotspot emphasis (Percent mode only; enforced above)
        if args.nt_hotspot_t is not None:
            gtmap_w = _transform_gtmap_robust_sigmoid(
                gtmap, t=float(args.nt_hotspot_t)
            ).astype(np.float32)
        else:
            gtmap_w = gtmap

        nt_weights = weights_tractogram * gtmap_w
        if strict_nan_inf:
            _check_no_nan_inf(nt_weights, f"nt_weights[{neurotrans}]")

        if args.NTmaps == "Percent":
            denom = float(np.sum(gtmap_w))
            if denom <= 0:
                _fail(
                    f"Denominator is zero for {neurotrans} (sum(transformed_gtmap)={denom}). "
                    "This can happen if filtering and/or hotspot transform removed all weight mass."
                )
            d[neurotrans] = float(np.sum(nt_weights)) / denom

        elif args.NTmaps == "Z":
            # Hotspot transform is forbidden in Z mode (enforced above).
            d[neurotrans] = float(np.sum(nt_weights))

        else:
            _fail(f"Unknown NTmaps value: {args.NTmaps!r} (expected 'Percent' or 'Z')")

    df = pd.DataFrame(d, index=[0])
    df.to_csv(out_NT_disc, index=False)
    print("Saved:", out_NT_disc)


if __name__ == "__main__":
    main()
