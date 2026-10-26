import os
import re
import nibabel as nib
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

def build_t1_candidates(sub_folder):
    """Return possible T1 filenames for one fMRI subject/session folder."""
    no_prefix = sub_folder.removeprefix('sub-')
    candidates = [
        f'm0wrp1{sub_folder}_T1w.nii',
        f'm0wrp1{no_prefix}_T1w.nii',
    ]

    # Example:
    #   fMRI folder: sub-NC001F12m
    #   T1 file:     m0wrp1NC001_12m_T1w.nii
    match = re.match(r'^(NC\d+)F(\d+)m$', no_prefix)
    if match:
        candidates.append(f'm0wrp1{match.group(1)}_{match.group(2)}m_T1w.nii')

    return list(dict.fromkeys(candidates))

def find_t1_file(t1_class_dir, sub_folder):
    for t1_fname in build_t1_candidates(sub_folder):
        t1_path = os.path.join(t1_class_dir, t1_fname)
        if os.path.isfile(t1_path):
            return t1_path
    return None

def find_classic_mat(sub_path, sub_folder):
    preferred = os.path.join(sub_path, f'{sub_folder}_Classic_results.mat')
    if os.path.isfile(preferred):
        return preferred

    for fname in sorted(os.listdir(sub_path)):
        if fname.endswith('_Classic_results.mat'):
            return os.path.join(sub_path, fname)
    return None

class PPMIDataset(Dataset):
    def __init__(self, fmri_root, t1_root):
        self.samples = []
        self.missing_t1 = []
        self.missing_fmri = []

        for class_name, label in [('PD', 1), ('NC', 0)]:
            fmri_class_dir = os.path.join(fmri_root, class_name)
            t1_class_dir = os.path.join(t1_root, class_name)
            if not os.path.isdir(fmri_class_dir) or not os.path.isdir(t1_class_dir):
                print(f'[Skip] {class_name}: directory not found')
                continue

            for sub_folder in sorted(os.listdir(fmri_class_dir)):
                sub_path = os.path.join(fmri_class_dir, sub_folder)
                if not os.path.isdir(sub_path):
                    continue

                mat_file = find_classic_mat(sub_path, sub_folder)
                if mat_file is None:
                    self.missing_fmri.append(sub_path)
                    continue

                t1_path = find_t1_file(t1_class_dir, sub_folder)
                if t1_path is None:
                    self.missing_t1.append((sub_folder, build_t1_candidates(sub_folder)))
                    continue

                self.samples.append((t1_path, mat_file, label, sub_folder))

        pd_count = sum(1 for _, _, label, _ in self.samples if label == 1)
        nc_count = sum(1 for _, _, label, _ in self.samples if label == 0)
        print(f'[PPMI] loaded {len(self.samples)} paired samples (PD={pd_count}, NC={nc_count})')

        if self.missing_fmri:
            print(f'[PPMI] missing fMRI Classic mat: {len(self.missing_fmri)}')
            for item in self.missing_fmri[:10]:
                print(f'  {item}')
        if self.missing_t1:
            print(f'[PPMI] missing T1 files: {len(self.missing_t1)}')
            for sub_folder, candidates in self.missing_t1[:10]:
                print(f'  {sub_folder}: {candidates}')
        if len(self.samples) == 0:
            raise RuntimeError('PPMI dataset is empty. Check PPMI roots and filename rules.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t1_path, mat_path, label, subject_id = self.samples[idx]
        t1_img = self._load_t1(t1_path)
        fmri_mat, adj = self._load_fmri(mat_path)
        label = torch.tensor(label, dtype=torch.long)
        return t1_img, fmri_mat, adj, label, subject_id

    def _load_t1(self, path):
        img_nii = nib.load(path)
        img = np.array(img_nii.get_fdata(), dtype=np.float32)
        if img.shape != (112, 112, 112):
            raise ValueError(f'Unexpected T1 shape: {img.shape}, file={path}')
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return torch.as_tensor(img, dtype=torch.float32)

    def _load_fmri(self, mat_path):
        try:
            mat = loadmat(mat_path)
            corr = mat['corr_matrix'].astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f'Failed to load .mat file: {mat_path}, {exc}') from exc

        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr[corr < 0] = 0.0
        np.fill_diagonal(corr, 1.0)
        corr = (corr + corr.T) / 2.0

        adj = (corr > 0.3).astype(np.int64)
        corr = np.ascontiguousarray(corr, dtype=np.float32)
        adj = np.ascontiguousarray(adj, dtype=np.int64)
        return torch.as_tensor(corr, dtype=torch.float32), torch.as_tensor(adj, dtype=torch.long)
