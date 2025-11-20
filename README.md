# MoCo-MP-qMRI
This repository contains the implementations of our manuscript "Motion-Compensated Implicit Neural Modeling for 3D Multiparametric Quantitative MRI".

We proposed a generalizable framework that integrates rapid navigator-based motion tracking with motion-informed implicit neural modeling for 3D multiparametric quantitative MRI. 

## Setup
1. Python 3.10.11
2. PyTorch 2.0.0
3. h5py, scipy, numpy, nibabel, tqdm
4. [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

## Files Description
```plaintext
SUMMIT/
├── run_demo.py              # Script for running the demo
├── model.pyc                # SUMMIT model
├── utils.pyc                # Supporting functions
├── README.md                # README file

Data/
├── rawdata.h5                 # Undersampled k-space data
├── imagedata_mask.mat         # Sampling mask
├── encTable.mat               # Sequential encoding table
├── estimate_motion_params.mat # Estimated motion parameters
├── mask.nii                   # Brain mask
├── gt/
│   ├── gt_T1.nii               # Ground truth of T1 map
│   ├── gt_T2.nii               # Ground truth of T2 map
│   ├── gt_T2star.nii           # Ground truth of T2star map
│   ├── gt_phi.nii              # Ground truth of phase map
├── recon/
│   ├── recon_T1.nii            # Motion-corrected Reconstruction of T1 map
│   ├── recon_T2.nii            # Motion-corrected Reconstruction of T2 map
│   ├── recon_T2star.nii        # Motion-corrected Reconstruction of T2star map
│   ├── recon_phi.nii           # Motion-corrected Reconstruction of phase map
├── corrupted/
│   ├── corrupted_T1.nii        # Motion-corrupted Reconstruction of T1 map
│   ├── corrupted_T2.nii        # Motion-corrupted Reconstruction of T2 map
│   ├── corrupted_T2star.nii    # Motion-corrupted Reconstruction of T2star map
│   ├── corrupted_phi.nii       # Motion-corrupted Reconstruction of phase map
```

## Usage
You can run "run_demo.py" to test the performance of motion correction.

Data for running the demo are available at [Google Drive](https://drive.google.com/drive/folders/1ih0OpCivzbg6NosKBKepkdnCvSn5PJ1Z?usp=sharing)
