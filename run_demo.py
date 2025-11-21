#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoCo-MP-qMRI Reconstruction Script
Author: Your Name
Description:
    This script performs motion-compensated quantitative MRI reconstruction.
"""

import os
import random
import argparse
import numpy as np
import torch
import h5py
import nibabel as nib
from scipy.io import loadmat, savemat
from model import MoCo
from utils import Motion_Cluster


DEFAULT_SEED = 1857

def set_random_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device(cuda_idx: int = 0) -> torch.device:
    """Select CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_idx}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device

def load_raw_kspace(filepath: str) -> torch.Tensor:
    """Load raw k-space from .h5 file."""
    with h5py.File(filepath, "r") as f:
        data = f["rawdata"][:]
    data = torch.tensor(data, dtype=torch.complex64)
    return data

def load_mask(filepath: str, device: torch.device) -> torch.Tensor:
    """Load k-space mask from .mat file and format it."""
    with h5py.File(filepath, "r") as f:
        mask = f["imagedata_mask"][:]

    mask = mask[..., 0].astype(bool)
    mask = torch.tensor(mask, dtype=torch.bool, device=device)

    # Expand dims to match expected input shape
    mask = mask.unsqueeze(0).unsqueeze(3).unsqueeze(-1)
    return mask

def load_enc_table(filepath: str) -> np.ndarray:
    """Load encoding table."""
    with h5py.File(filepath, "r") as f:
        enc_table = f["encTable"][:]
    return enc_table

def build_arg_parser():
    parser = argparse.ArgumentParser(description='MoCo-MP-qMRI Reconstruction')

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)

    # WNNM and SC regularization settings
    parser.add_argument('--denoise_start', type=int, default=50)
    parser.add_argument('--A_w', type=float, default=0.05)
    parser.add_argument('--T1_w', type=float, default=0.2)
    parser.add_argument('--T2_w', type=float, default=2.0)
    parser.add_argument('--T2star_w', type=float, default=2.0)
    parser.add_argument('--phi_w', type=float, default=0.2)
    parser.add_argument('--SC_w', type=float, default=1e-3)

    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--cuda_idx', type=int, default=1)

    return parser

def save_results(mat_dir: str, nii_dir: str, voxel_size, A, T1, T2, T2star, phi, csm):
    """Save .mat and .nii results."""
    os.makedirs(mat_dir, exist_ok=True)
    os.makedirs(nii_dir, exist_ok=True)

    to_save = {
        "pre_A.mat": A,
        "pre_T1.mat": T1,
        "pre_T2.mat": T2,
        "pre_T2star.mat": T2star,
        "pre_phi.mat": phi,
        "pre_csm.mat": csm,
    }

    for name, val in to_save.items():
        savemat(os.path.join(mat_dir, name), {name[:-4]: val})

    # NIfTI results
    save_nii(A, os.path.join(nii_dir, "pre_A.nii"), voxel_size)
    save_nii(T1, os.path.join(nii_dir, "pre_T1.nii"), voxel_size)
    save_nii(T2, os.path.join(nii_dir, "pre_T2.nii"), voxel_size)
    save_nii(T2star, os.path.join(nii_dir, "pre_T2star.nii"), voxel_size)
    save_nii(phi, os.path.join(nii_dir, "pre_phi.nii"), voxel_size)
    save_nii(np.abs(csm), os.path.join(nii_dir, "pre_csm_mag.nii"), voxel_size)

    print("Results saved.")

def save_nii(arr, path, voxel_size=[1,1,1]):
    affine=np.array([[voxel_size[0],0,0,0],
                     [0,voxel_size[1],0,0],
                     [0,0,voxel_size[2],0],
                     [0,0,0,1]])
    nib.Nifti1Image(arr,affine).to_filename(path)
    return None

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = get_device(args.cuda_idx)

    data_root = "/home/gylao/lab/MultiContrast/src/Motion/demo_code_MoCo-MP-qMRI/data"
    raw_ksp_path = os.path.join(data_root, "rawdata.h5")
    mask_path = os.path.join(data_root, "imagedata_mask.mat")
    enc_table_path = os.path.join(data_root, "encTable.mat")
    motion_path = os.path.join(data_root, "estimate_motion_params.mat")

    # Load Data
    ksp = load_raw_kspace(raw_ksp_path)
    ksp_mask = load_mask(mask_path, device)
    enc_table = load_enc_table(enc_table_path)

    motion_params = loadmat(motion_path)["estimate_motion_params"]
    best_k, grouped_motion_params, motion_index = Motion_Cluster(motion_params, max_cluster=10)

    # Imaging parameters
    TR = 0.020
    FA = 8 / 180 * np.pi
    Nrd, Npe, Nspe, Nchl, Ntau, Necho = ksp.shape
    Tn = np.arange(1, ksp_mask.shape[4] + 1)
    tau = np.array([0.025, 0.050, 0.070, 0.090])
    TE = np.array([0.00380, 0.00943, 0.01506])
    voxel_size = [1, 1, 3]

    # Reconstruction model
    model = MoCo(ksp, grouped_motion_params, motion_index, enc_table, ksp_mask,
        TR, FA, Tn, tau, TE,
        voxel_size, device, args
    )

    pre_A, pre_T1, pre_T2, pre_T2star, pre_phi, csm = model.Recon()

    # Save outputs
    out_root = os.path.join(data_root, 'test_demo')
    save_results(
        os.path.join(out_root, 'mat'),
        os.path.join(out_root, 'nii'),
        voxel_size,
        pre_A, pre_T1, pre_T2, pre_T2star, pre_phi, csm
    )


if __name__ == '__main__':
    main()
