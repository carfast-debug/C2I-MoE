import os
import numpy as np
import torch
import math
from torch.utils.data import Dataset
from scipy.io import loadmat
import nibabel as nib
import re
from collections import defaultdict


def extract_subject_id_from_t1(t1_path):
    fname = os.path.basename(t1_path)
    match = re.search(r'(PD|NC)\d+', fname)
    if match:
        return match.group(0)
    return fname


def group_pairs_by_subject(pairs):
    subject_groups = defaultdict(list)
    for t1_path, mat_path, label in pairs:
        subject_id = extract_subject_id_from_t1(t1_path)
        subject_groups[subject_id].append((t1_path, mat_path, label))
    return subject_groups


def parse_fmri_subid(folder_name):
    pattern = r'^sub-(PD|NC)(\d+)(F(\d+m))?$'
    match = re.match(pattern, folder_name)
    if not match:
        return None, None, None
    group = match.group(1)
    num = match.group(2)
    time_suffix = match.group(4)
    return group, num, time_suffix


def find_t1_file(t1_dir, group, num, time_suffix):
    kk = group
    candidates = []
    if time_suffix:
        candidates.append(f"m0wrp1{kk}{num}_{time_suffix}_T1w.nii")
    candidates.append(f"m0wrp1{kk}{num}_T1w.nii")
    for fname in candidates:
        fpath = os.path.join(t1_dir, fname)
        if os.path.isfile(fpath):
            return fpath
    return None


def create_paired_samples(fmri_root, t1_root):
    pairs = []
    for class_name, label in [('NC', 0), ('PD', 1)]:
        fmri_class_dir = os.path.join(fmri_root, class_name)
        t1_class_dir = os.path.join(t1_root, class_name)
        if not os.path.isdir(fmri_class_dir) or not os.path.isdir(t1_class_dir):
            print(f"跳过 {class_name}: 目录不存在")
            continue
        for sub_folder in os.listdir(fmri_class_dir):
            sub_path = os.path.join(fmri_class_dir, sub_folder)
            if not os.path.isdir(sub_path):
                continue
            group, num, time_suffix = parse_fmri_subid(sub_folder)
            if group is None:
                continue
            t1_path = find_t1_file(t1_class_dir, group, num, time_suffix)
            if t1_path is None:
                continue
            mat_file = None
            for f in os.listdir(sub_path):
                if f.endswith('_Classic_results.mat'):
                    mat_file = os.path.join(sub_path, f)
                    break
            if mat_file is None:
                for f in os.listdir(sub_path):
                    if f.endswith('_Slow4_results.mat'):
                        mat_file = os.path.join(sub_path, f)
                        break
            if mat_file is None:
                continue
            pairs.append((t1_path, mat_file, label))
    return pairs


class PairedDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t1_path, mat_path, label = self.samples[idx]
        t1_img = self._load_t1(t1_path)          # (1, 112, 112, 112) tensor
        fmri_mat, adj = self._load_fmri(mat_path) # (116,116) tensors

        if self.augment:
            # ---------- T1 数据增强 ----------
            # 随机水平翻转（左右）
            if np.random.rand() > 0.5:
                t1_img = torch.flip(t1_img, dims=[2])  # 沿 H 轴翻转

            # 随机旋转（绕深度轴 Z，范围 ±10°）
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10) * math.pi / 180.0
                rot_mat_3x3 = torch.tensor([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle),  math.cos(angle), 0],
                    [0, 0, 1]
                ], dtype=torch.float32)
                rot_mat_3x4 = torch.zeros(3, 4)
                rot_mat_3x4[:3, :3] = rot_mat_3x3
                rot_mat = rot_mat_3x4.unsqueeze(0)  # (1, 3, 4)
                # 添加 batch 维
                img_in = t1_img.unsqueeze(0)  # (1, 1, 112, 112, 112)
                grid = torch.nn.functional.affine_grid(rot_mat, img_in.size(), align_corners=False)
                t1_img = torch.nn.functional.grid_sample(img_in, grid, align_corners=False).squeeze(0)

            # 随机缩放（0.9 ~ 1.1）
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                new_size = int(112 * scale)
                # 先缩放到 new_size，再插值回 112
                img_in = t1_img.unsqueeze(0)
                img_scaled = torch.nn.functional.interpolate(
                    img_in, size=(new_size, new_size, new_size), mode='trilinear', align_corners=False)
                img_scaled = torch.nn.functional.interpolate(
                    img_scaled, size=(112, 112, 112), mode='trilinear', align_corners=False)
                t1_img = img_scaled.squeeze(0)

            # ---------- fMRI 数据增强 ----------
            # 添加高斯噪声（标准差 0.05）
            if np.random.rand() > 0.5:
                noise = torch.randn_like(fmri_mat) * 0.05
                fmri_mat = fmri_mat + noise
                fmri_mat = torch.clamp(fmri_mat, 0.0, 1.0)

            # 随机丢弃边（10% 概率将连接置 0）
            if np.random.rand() > 0.5:
                dropout_mask = (torch.rand_like(fmri_mat) > 0.1).float()
                fmri_mat = fmri_mat * dropout_mask

            # 增强后重新对称化并保持对角线为 1
            fmri_mat = (fmri_mat + fmri_mat.transpose(0, 1)) / 2.0
            fmri_mat.fill_diagonal_(1.0)
            adj = (fmri_mat > 0.3).long()  # 重新计算邻接矩阵

        label = torch.tensor(label, dtype=torch.long)
        return t1_img, fmri_mat, adj, label

    def _load_t1(self, path):
        img_nii = nib.load(path)
        img = np.array(img_nii.get_fdata(), dtype=np.float32)
        assert img.shape == (112, 112, 112), f"T1图像形状错误: {img.shape}"
        img = np.expand_dims(img, axis=0)  # (1, 112, 112, 112)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return torch.as_tensor(img, dtype=torch.float32)

    def _load_fmri(self, mat_path):
        try:
            mat = loadmat(mat_path)
            corr = mat['corr_matrix'].astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"加载.mat文件失败: {mat_path}, {e}")
        if np.any(np.isnan(corr)):
            corr = np.nan_to_num(corr, nan=0.0)
        if np.any(np.isinf(corr)):
            corr = np.nan_to_num(corr, posinf=0.0, neginf=0.0)
        corr[corr < 0] = 0.0
        np.fill_diagonal(corr, 1.0)
        corr = (corr + corr.T) / 2.0

        adj = (corr > 0.3).astype(np.int64)
        corr = np.ascontiguousarray(corr, dtype=np.float32)
        adj = np.ascontiguousarray(adj, dtype=np.int64)
        return torch.as_tensor(corr, dtype=torch.float32), torch.as_tensor(adj, dtype=torch.long)
