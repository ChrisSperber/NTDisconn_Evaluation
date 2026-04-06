#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
import argparse
import nibabel as nib
import numpy as np
from dipy.tracking._utils import _mapping_to_voxel, _to_voxel_coordinates
import subprocess
from scipy.stats import zscore
import os
import ants
from tqdm import tqdm
import pandas as pd
import requests
import shutil

def buildArgsParser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    #p.add_argument('in_neurotrans',
    #               help='Input Neurotransmitter system (Specific name can be found in HCP_NT folder')
    p.add_argument('ID',
                   help='Subject ID')
    p.add_argument('in_lesion',
                   help='Input individual Lesionmask in MNI152 (1mm iso)')
    p.add_argument('output_dir',
                   help='Specify output directory')
    p.add_argument('--discStreamlines', default='y',
                   help='Create disconnected streamline output? [y|n]')
    p.add_argument('--NTmaps', default='Percent',
                   help='Which NT maps to use? [Z|Percent]')
    p.add_argument('--filter', default='n',
                   help='Filter Streamlines - enter percentile [y|n]')

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
    p.add_argument('--nt_hotspot_t', type=float, default=None,
                   help=('Optional robust-sigmoid hotspot transform for gtmap weights. '
                         'Provide threshold t (float) in robust-z units. '
                         'Examples: 1.0 (mild), 1.5 (moderate), 2.0 (strong). '
                         'Only valid with --NTmaps Percent; will error for --NTmaps Z.'))

    # NEW: subtract global disconnection burden (F) from each NT percent score
    #
    # In Percent mode, the base score is:
    #   P_k = sum(disconnected * w_k) / sum(w_k)
    #
    # where disconnected is the 0/1 streamline disconnection mask.
    #
    # The global disconnection fraction is:
    #   F = (# disconnected streamlines) / (total # streamlines)
    #
    # If enabled, the script outputs:
    #   P_k - F
    #
    # This removes the shared component driven purely by "how many streamlines are disconnected".
    # Note: values can become negative.
    p.add_argument('--subtract_disc_fraction', default='n',
                   help='Subtract global disconnection fraction F from each NT Percent score [y|n]. '
                        'Only valid with --NTmaps Percent.')

    return p


def _transform_gtmap_robust_sigmoid(gtmap, t, alpha=3.0, eps=1e-12, zero_eps=0.0):
    """
    Robust hotspot emphasis for sparse NT weights (many zeros + tail).

    Key idea:
      - If many weights are exactly 0, MAD around the global median can be 0.
      - In that case, standardize using the non-zero subset only.
      - Keep zeros exactly zero after transform.

    Returns weights in [0,1], suitable for proportion scoring:
        sum(disconnected*w) / sum(w)
    """
    x = np.asarray(gtmap, dtype=np.float64)

    nz = x > float(zero_eps)
    if np.sum(nz) == 0:
        raise RuntimeError("All NT weights are zero; cannot apply hotspot transform.")

    x_nz = x[nz]
    med = np.median(x_nz)
    mad = np.median(np.abs(x_nz - med))
    scale = 1.4826 * mad

    if (not np.isfinite(scale)) or (scale < eps):
        q25, q75 = np.percentile(x_nz, [25, 75])
        scale = float(q75 - q25)

    if (not np.isfinite(scale)) or (scale < eps):
        raise RuntimeError("Cannot apply hotspot transform because non-zero NT weights show ~0 dispersion.")

    z = (x_nz - med) / scale
    w_nz = 1.0 / (1.0 + np.exp(-float(alpha) * (z - float(t))))

    w = np.zeros_like(x, dtype=np.float64)
    w[nz] = w_nz

    if np.sum(w) <= 0:
        raise RuntimeError("Hotspot-transformed weights have zero sum.")

    return(w)


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    reference = "HCPA422-T1w-500um-norm.nii.gz"
    out_NT_disc = os.path.join(args.output_dir, args.ID +"_NT_Diconnect_"+args.NTmaps+".csv")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Enforce: hotspot transform only supported in Percent mode
    if args.nt_hotspot_t is not None and args.NTmaps != 'Percent':
        raise RuntimeError("--nt_hotspot_t is only implemented for --NTmaps Percent.")

    # Enforce: subtract F only supported in Percent mode
    if args.subtract_disc_fraction == 'y' and args.NTmaps != 'Percent':
        raise RuntimeError("--subtract_disc_fraction is only implemented for --NTmaps Percent.")

    def define_streamlines(streamlines, lesion, reference):#, NT_weights_SUM):#, weights):
        metric_tractrogram = []
        metric_tractogram_preserved = []

        les = lesion.get_fdata()
        affine = reference.affine
        lin_T, offset = _mapping_to_voxel(affine)

        for s in tqdm(range(2000000), desc="Evaluate streamlines"):
            streamline = streamlines[s]
            ### location
            x_ind_2 = _to_voxel_coordinates(streamline[:], lin_T, offset)[:, 0]
            y_ind_2 = _to_voxel_coordinates(streamline[:], lin_T, offset)[:, 1]
            z_ind_2 = _to_voxel_coordinates(streamline[:], lin_T, offset)[:, 2]

            if np.sum(les[x_ind_2, y_ind_2, z_ind_2]) > 0:
                metric_tractrogram.append(1)
                #metric_tractogram_preserved.append(0)
            else:
                metric_tractrogram.append(0)
            #    metric_tractogram_preserved.append(a[s])

        return(metric_tractrogram)

    tck_file = "HCP422_2_million.tck"
    if os.path.isfile(tck_file):
        print("Tactogram exists")
    else:
        print("Downloading Tractogram...........")
        osf_url = "https://osf.io/download/nduwc/"
        response = requests.get(osf_url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(tck_file, "wb") as file, tqdm(desc=tck_file, total=total_size, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))
        print("Download complete!")
       
    ### Create Warpfield for MNI to HCPA transformation
    if os.path.isfile('MNI_to_HCPA_Warp.nii.gz'):
        print("Coregistration MNI to HCPA done")
    else:
        print("Coregistration MNI to HCPA .....")
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = "1"
        mi = ants.image_read("MNI152_T1_1mm.nii.gz")
        fi = ants.image_read(reference)
        tx = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', random_seed=1, singleprecision=False)
        forwardtrans = tx['fwdtransforms']
        shutil.copyfile(forwardtrans[1], "MNI_to_HCPA.mat")
        shutil.copyfile(forwardtrans[0], "MNI_to_HCPA_Warp.nii.gz")
        print("Coregistration done!")

        
    out_weights_tractogram_disc = os.path.join(args.output_dir,
                                                   args.ID + "_Disc_Streamlines.txt")
    
    if os.path.isfile(out_weights_tractogram_disc) == True:
        print("disc sl already calculated")
        weights_tractogram = np.loadtxt(out_weights_tractogram_disc)
    else:

        print("Loading streamlines ##########################################")
        tractogram = nib.streamlines.load(tck_file)
        streamlines = tractogram.streamlines
        header_sl = tractogram.header

        ### Bring Input Lesion (MNI) in HCPA MNI
        standard = "HCPA422-T1w-500um-norm.nii.gz"
        listtransf = ['MNI_to_HCPA_Warp.nii.gz', "MNI_to_HCPA.mat"]
        fi = ants.image_read(standard)
        movmap = ants.image_read(args.in_lesion)
        mywarpedimage = ants.apply_transforms(fixed=fi, moving=movmap,
                                              transformlist=listtransf, interpolator='multiLabel')

        output = mywarpedimage.numpy()
        ref = nib.load(standard)
        lesion = nib.Nifti1Image(output, ref.affine, ref.header)
        nib.save(lesion, "tmp_les.nii.gz")

        weights_tractogram = define_streamlines(streamlines, lesion, nib.load(reference))

        
        if args.discStreamlines == 'y':
            np.savetxt(out_weights_tractogram_disc, weights_tractogram)

        
    if os.path.isfile(out_NT_disc) == True:
        print("NT Disc already calculated")
    else:
    
        d = {}
        d["ID"] = args.ID
        d["Disc_SL"] = np.sum(weights_tractogram)

        # NEW: global disconnection fraction F
        F = float(np.sum(weights_tractogram)) / float(len(weights_tractogram))
        d["Disc_Frac"] = F

        print("Evaluate NT systems......................")
        for neurotrans in ["5HT1a_way_hc36_savli", "5HT1b_p943_hc65_gallezot", "5HT2a_cimbi_hc29_beliveau", "5HT4_sb20_hc59_beliveau", "5HT6_gsk_hc30_radhakrishnan", "5HTT_dasb_hc100_beliveau", "D1_SCH23390_hc13_kaller", "D2_flb457_hc37_smith", "DAT_fpcit_hc174_dukart_spect", "A4B2_flubatine_hc30_hillmer", "VAChT_feobv_hc18_aghourian_sum", "mGluR5_abp_hc22_rosaneto", "GABAa-bz_flumazenil_hc16_norgaard", "NAT_MRB_hc77_ding", "H3_cban_hc8_gallezot", "M1_lsn_hc24_naganawa", "CB1_omar_hc77_normandin", "NMDA_ge179_hc29_galovic", "MU_carfentanil_hc204_kantonen"]:
            print(neurotrans)
            in_neurotrans_weights = os.path.join("HCP_NT", args.NTmaps, neurotrans,
                                                 "GT_" + neurotrans + "_weights_disc_Tractogram.txt")
            out_connect = os.path.join(args.output_dir, args.ID + "_" + neurotrans + "_Diconnectome.csv")
            #out_connect_pres = os.path.join(args.output_dir, args.in_neurotrans + "_Preserved_Connectome.csv")
            
            
            gtmap = np.loadtxt(in_neurotrans_weights)

            ####### filter by percentile ######
            if args.filter != 'n':
                cutoff = cutoff = np.percentile(gtmap, 75)
                gtmap[gtmap<cutoff] = 0
            ###################################

            # NEW: optional hotspot emphasis for Percent mode
            if args.nt_hotspot_t is not None:
                gtmap_used = _transform_gtmap_robust_sigmoid(gtmap, args.nt_hotspot_t)
            else:
                gtmap_used = gtmap

            nt_weights = weights_tractogram * gtmap_used
            np.savetxt("tmp_disc.txt", nt_weights)

            if args.NTmaps == 'Percent':
                score = np.sum(nt_weights) / np.sum(gtmap_used)

                # NEW: optional subtraction of global disconnection fraction
                if args.subtract_disc_fraction == 'y':
                    score = score - F

                d[neurotrans] = score

            if args.NTmaps == 'Z':
                d[neurotrans] = np.sum(nt_weights)

            '''
            if args.Connectome == 'y':
                print("Creating Connectomes #########################")
                #tck2connectome_cmd = "tck2connectome HCP422_2_million.tck BN_Atlas_277_05mm_HCPA.nii.gz " + out_connect_pres + " -tck_weights_in "+out_weights_tractogram_pres+" --assignment_radial_search 4 -symmetric -f"  # -zero_diagonal
                #subprocess.call(tck2connectome_cmd, shell=True)

                tck2connectome_cmd = "tck2connectome HCP422_2_million.tck BN_Atlas_277_05mm_HCPA.nii.gz " + out_connect + " -tck_weights_in tmp_disc.txt --assignment_radial_search 4 -symmetric -f"  # -zero_diagonal
                subprocess.call(tck2connectome_cmd, shell=True)
            '''
        
        df = pd.DataFrame(d, index=[0])
        df.to_csv(out_NT_disc)

if __name__ == "__main__":
    main()

# %%