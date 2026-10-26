import os
import csv
import numpy as np
from collections import defaultdict
from paired_dataset import (
    create_paired_samples, group_pairs_by_subject, extract_subject_id_from_t1
)

def split_and_save(fmri_root, t1_root, output_dir='divide',
                   train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                   val_pd_ratio=0.8, test_pd_ratio=0.8,
                   required_train_subjects=None, seed=42):
    """
    按患者划分数据：
      - 训练集、验证集、测试集占总体的比例分别为 train_ratio, val_ratio, test_ratio
      - 验证集和测试集内部 PD:NC 比例由 val_pd_ratio 和 test_pd_ratio 控制
      - 训练集必须包含 required_train_subjects 列表中的受试者
      - 训练集包含剩余所有受试者
    """
    if required_train_subjects is None:
        required_train_subjects = []

    all_pairs = create_paired_samples(fmri_root, t1_root)
    subject_groups = group_pairs_by_subject(all_pairs)
    subject_ids = list(subject_groups.keys())
    subject_labels = [subject_groups[s][0][2] for s in subject_ids]  # 0=NC, 1=PD

    # 按标签分组
    nc_ids = [s for s, lbl in zip(subject_ids, subject_labels) if lbl == 0]
    pd_ids = [s for s, lbl in zip(subject_ids, subject_labels) if lbl == 1]
    total_nc = len(nc_ids)
    total_pd = len(pd_ids)
    total = total_nc + total_pd

    # 验证集和测试集各自需要的总人数
    val_total = int(total * val_ratio)
    test_total = int(total * test_ratio)

    # 验证集中 PD 和 NC 人数（根据比例计算，确保整数）
    val_pd_needed = int(val_total * val_pd_ratio)
    val_nc_needed = val_total - val_pd_needed
    # 测试集
    test_pd_needed = int(test_total * test_pd_ratio)
    test_nc_needed = test_total - test_pd_needed

    # 检查是否有足够人数
    if val_pd_needed > total_pd:
        raise ValueError(f"验证集需要 {val_pd_needed} 个 PD，但总共只有 {total_pd} 个 PD")
    if test_pd_needed > total_pd - val_pd_needed:
        raise ValueError(f"测试集需要 {test_pd_needed} 个 PD，剩余 PD 不足")
    if val_nc_needed > total_nc:
        raise ValueError(f"验证集需要 {val_nc_needed} 个 NC，但总共只有 {total_nc} 个 NC")
    if test_nc_needed > total_nc - val_nc_needed:
        raise ValueError(f"测试集需要 {test_nc_needed} 个 NC，剩余 NC 不足")

    # 检查 required_train_subjects 是否都在 NC 中且存在
    required_set = set(required_train_subjects)
    missing_req = required_set - set(nc_ids)
    if missing_req:
        raise ValueError(f"要求的训练集受试者 {missing_req} 不在 NC 列表中")

    # 随机打乱（确保 required 受试者不会被选入验证/测试集）
    np.random.seed(seed)
    # 从 nc_ids 中移除 required 受试者
    nc_ids_pool = [s for s in nc_ids if s not in required_set]
    np.random.shuffle(nc_ids_pool)
    np.random.shuffle(pd_ids)

    # 分配验证集和测试集
    val_nc = nc_ids_pool[:val_nc_needed]
    remaining_nc_after_val = nc_ids_pool[val_nc_needed:]
    test_nc = remaining_nc_after_val[:test_nc_needed]
    train_nc = remaining_nc_after_val[test_nc_needed:] + list(required_set)  # 强制包含的加入训练集

    val_pd = pd_ids[:val_pd_needed]
    remaining_pd_after_val = pd_ids[val_pd_needed:]
    test_pd = remaining_pd_after_val[:test_pd_needed]
    train_pd = remaining_pd_after_val[test_pd_needed:]

    # 合并
    train_ids = train_nc + train_pd
    val_ids = val_nc + val_pd
    test_ids = test_nc + test_pd

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存 CSV
    csv_path = os.path.join(output_dir, f'data_split_seed{seed}_balanced_val_test.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject_id', 'set'])
        for s in train_ids:
            writer.writerow([s, 'train'])
        for s in val_ids:
            writer.writerow([s, 'val'])
        for s in test_ids:
            writer.writerow([s, 'test'])

    # 统计并打印
    print(f"数据划分完成 (训练:{train_ratio:.0%}, 验证:{val_ratio:.0%}, 测试:{test_ratio:.0%})")
    print(f"训练集: NC={len(train_nc)}, PD={len(train_pd)} → 总 {len(train_ids)} 人 (PD比例={len(train_pd)/(len(train_pd)+len(train_nc)):.1%})")
    print(f"验证集: NC={len(val_nc)}, PD={len(val_pd)} → 总 {len(val_ids)} 人 (PD比例={len(val_pd)/(len(val_pd)+len(val_nc)):.1%})")
    print(f"测试集: NC={len(test_nc)}, PD={len(test_pd)} → 总 {len(test_ids)} 人 (PD比例={len(test_pd)/(len(test_pd)+len(test_nc)):.1%})")
    print(f"划分结果保存至 {csv_path}")

    # 返回样本对列表
    train_pairs = []
    for s in train_ids:
        train_pairs.extend(subject_groups[s])
    val_pairs = []
    for s in val_ids:
        val_pairs.extend(subject_groups[s])
    test_pairs = []
    for s in test_ids:
        test_pairs.extend(subject_groups[s])

    return train_pairs, val_pairs, test_pairs
