"""C2I-MoE training loop; use train.py to supply paths and run settings."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import math
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from torch.amp import GradScaler, autocast

from paired_dataset import (
    create_paired_samples, PairedDataset,
    group_pairs_by_subject, extract_subject_id_from_t1
)
from fusion_model import T1FeatureExtractor, FMRIFeatureExtractor, I2MoE_Fusion
from split_dataset import split_and_save
from utils.rotation import compute_rotation_matrix
from utils.lr_decay import param_groups_lrd
import warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", message=".*TypedStorage.*")
warnings.filterwarnings("ignore", message=".*Converting mask.*")

# ======================== RollingQ 超参数 ========================
MAX_ROTATIONS = 2
BETA_SYN = 0.1
RHO = 1.0

# ======================== 固定超参数 ========================
FMRI_ROOT = None  # Set by train.py.
T1_ROOT = None  # Set by train.py.
T1_PRETRAIN = None  # Set by train.py.
FMRI_PRETRAIN = None  # Set by train.py.

# Cross-modal contrastive learning hyperparameters.
# CONTRAST_LOSS_WEIGHT controls how much the auxiliary T1-fMRI alignment loss
# contributes to total_loss. The released K configuration uses 0.02.
# CONTRAST_TEMPERATURE controls the sharpness of the InfoNCE similarity logits.
INIT_CHECKPOINT = None
SELECTION_SCORE = "auc-f1"
CONTRAST_LOSS_WEIGHT = 0.02
CONTRAST_TEMPERATURE = 0.10

T1_LR = 5e-7          # 仅作为占位，实际会被分层策略覆盖
FMRI_LR = 5e-7
EXPERT_LR = 1e-4
BATCH_SIZE = 28
MIN_LR = 1e-7
WARMUP_EPOCHS = 30
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 60
USE_AMP = False
NUM_WORKERS = 8

FREEZE_T1 = False          # 3DINO 可训练（分层学习率）
FREEZE_FMRI = True         # 冻结 BrainGFM 骨干，仅微调 GraphPrompt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestPairedDataset(PairedDataset):
    """Attach subject IDs for test-error records; supports spawned workers."""

    def __getitem__(self, idx):
        t1_img, fmri_mat, adj, label = super().__getitem__(idx)
        t1_path, _, _ = self.samples[idx]
        subject_id = extract_subject_id_from_t1(t1_path)
        return t1_img, fmri_mat, adj, label, subject_id


def run_experiment(data_split_seed, exp_seed, interaction_loss_weight):
    """单次实验，支持指定交互损失权重"""
    print(f"\n{'='*60}")
    print(f"开始实验: DATA_SPLIT_SEED={data_split_seed}, EXP_SEED={exp_seed}, WEIGHT={interaction_loss_weight}")
    print(f"使用设备: {device} | 模型选择: {SELECTION_SCORE}")
    print(f"{'='*60}")

    np.random.seed(exp_seed)
    torch.manual_seed(exp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(exp_seed)

    # 数据准备
    print("正在配对并划分数据...")
    train_pairs, val_pairs, test_pairs = split_and_save(
        FMRI_ROOT, T1_ROOT, output_dir='divide', seed=data_split_seed
    )
    print(f"训练集样本数: {len(train_pairs)}, 验证集: {len(val_pairs)}, 测试集: {len(test_pairs)}")

    train_dataset = PairedDataset(train_pairs, augment=True)
    val_dataset = PairedDataset(val_pairs, augment=False)
    test_dataset = TestPairedDataset(test_pairs, augment=False)

    # 统计训练集类别样本数（用于加权损失）
    print("正在统计训练集类别样本数...")
    train_labels_list = [label for _, _, _, label in train_dataset]
    train_nc = train_labels_list.count(0)
    train_pd = train_labels_list.count(1)
    total_train = train_nc + train_pd
    nc_weight = total_train / (2 * train_nc) if train_nc > 0 else 1.0
    pd_weight = total_train / (2 * train_pd) if train_pd > 0 else 1.0
    print(f"训练集样本分布: NC={train_nc}, PD={train_pd} (总={total_train})")
    print(f"计算得到的类别权重: NC={nc_weight:.4f}, PD={pd_weight:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=NUM_WORKERS)

    # 模型构建
    t1_extractor = T1FeatureExtractor(pretrained_path=T1_PRETRAIN)
    fmri_extractor = FMRIFeatureExtractor(pretrained_path=FMRI_PRETRAIN)

    # 详细打印 3DINO 权重加载情况（此处省略详细输出，保持简洁）
    print("\n" + "=" * 50)
    print("详细 3DINO 权重加载分析")
    print("=" * 50)
    state_dict = torch.load(T1_PRETRAIN, map_location='cpu')
    if 'teacher' in state_dict:
        pretrained = state_dict['teacher']
    else:
        pretrained = state_dict
    model_dict = t1_extractor.backbone.state_dict()
    new_pretrained = {}
    for k, v in pretrained.items():
        if k.startswith('backbone.'):
            new_k = k[9:]
        else:
            new_k = k
        new_pretrained[new_k] = v
    missing_keys = []
    unexpected_keys = []
    for k in model_dict:
        if k not in new_pretrained:
            missing_keys.append(k)
        else:
            if model_dict[k].shape != new_pretrained[k].shape:
                missing_keys.append(f"{k} (shape mismatch: {model_dict[k].shape} vs {new_pretrained[k].shape})")
    for k in new_pretrained:
        if k not in model_dict:
            unexpected_keys.append(k)
    print(f"Missing keys ({len(missing_keys)} in total):")
    if missing_keys:
        for k in missing_keys[:20]:
            print(f"  {k}")
    else:
        print("  None")
    print(f"Unexpected keys ({len(unexpected_keys)} in total):")
    if unexpected_keys:
        for k in unexpected_keys[:20]:
            print(f"  {k}")
    else:
        print("  None")
    print("=" * 50 + "\n")

    # 检测 3DINO 块结构
    print("\n" + "=" * 50)
    print("检测 3DINO 预训练权重加载情况")
    print("=" * 50)
    checkpoint = torch.load(T1_PRETRAIN, map_location='cpu')
    if 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    else:
        state_dict = checkpoint
    pretrained_block_indices = set()
    for key in state_dict.keys():
        if 'blocks' in key:
            parts = key.split('.')
            try:
                idx = parts.index('blocks')
                chunk_idx = int(parts[idx + 1])
                block_idx_in_chunk = int(parts[idx + 2])
                pretrained_block_indices.add((chunk_idx, block_idx_in_chunk))
            except (IndexError, ValueError):
                continue
    expected_block_pairs = set()
    chunk_size = 6
    for chunk in range(4):
        for inner in range(chunk_size):
            expected_block_pairs.add((chunk, inner))
    print(f"预训练权重中检测到的 (chunk, inner) 块对数量: {len(pretrained_block_indices)} (预期 {len(expected_block_pairs)})")
    if len(pretrained_block_indices) == len(expected_block_pairs):
        print("✓ 预训练权重包含完整的 24 个 Transformer 块（4 个 chunk，每个 chunk 6 个块）")
        print(f"  实际索引示例: {sorted(pretrained_block_indices)[:10]} ...")
    else:
        print("✗ 预训练权重中块数量不足或过多，请检查权重文件！")
        print(f"  实际索引: {sorted(pretrained_block_indices)[:10]} ...")
    current_model = t1_extractor.backbone
    if hasattr(current_model, 'blocks') and isinstance(current_model.blocks, nn.ModuleList):
        real_blocks = 0
        for blk_chunk in current_model.blocks:
            for subblk in blk_chunk:
                if not isinstance(subblk, nn.Identity):
                    real_blocks += 1
        print(f"当前模型中的真实 Transformer 块数量: {real_blocks}")
        if real_blocks == 24:
            print("✓ 当前模型包含正确的 24 个 Transformer 块")
        else:
            print(f"✗ 当前模型块数量 {real_blocks} 与预期 24 不符！")
    else:
        print("无法获取当前模型块数量，请手动检查。")
    print("\n预训练权重键名示例（前10个与 blocks 相关的）:")
    block_keys = [k for k in state_dict.keys() if 'blocks' in k]
    for k in block_keys[:10]:
        print(f"  {k}")
    print("\n当前模型参数名示例（前10个与 blocks 相关的）:")
    model_block_keys = [name for name, _ in current_model.named_parameters() if 'blocks' in name]
    for k in model_block_keys[:10]:
        print(f"  {k}")
    print("=" * 50 + "\n")

    # 创建模型，传入交互损失权重
    model = I2MoE_Fusion(
        t1_extractor, fmri_extractor,
        hidden_size=768, num_classes=2, dropout=0.5,
        interaction_loss_weight=interaction_loss_weight,
        temperature=2,
        num_layers=3, gated_layers=[2],
        contrast_dim=128, contrast_temperature=CONTRAST_TEMPERATURE
    )
    if INIT_CHECKPOINT:
        checkpoint = torch.load(INIT_CHECKPOINT, map_location="cpu", weights_only=True)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)

    # 冻结策略
    if FREEZE_T1:
        for param in model.t1_extractor.parameters():
            param.requires_grad = False
        print("\n[冻结] 3DINO 骨干网络已冻结")
    else:
        print("\n[提示] 3DINO 骨干将使用分层学习率衰减（可训练）")
        t1_param_groups = param_groups_lrd(
            model.t1_extractor.backbone,
            weight_decay=WEIGHT_DECAY,
            no_weight_decay_list=[],
            layer_decay=0.75
        )
        T1_BASE_LR = 5e-6
        for group in t1_param_groups:
            scale = group['lr_scale']
            group['lr'] = T1_BASE_LR * scale
            del group['lr_scale']
        print(f"[3DINO] 分层学习率范围: min={min(g['lr'] for g in t1_param_groups):.2e}, max={max(g['lr'] for g in t1_param_groups):.2e}")

    if FREEZE_FMRI:
        for name, param in model.fmri_extractor.backbone.named_parameters():
            if 'graph_prompt' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("[冻结] BrainGFM 骨干已冻结（仅 GraphPrompt 可训练）")

    # 优化器分组
    graph_prompt_param_names = set()
    for name, param in model.named_parameters():
        if 'graph_prompt' in name and param.requires_grad:
            graph_prompt_param_names.add(name)
    graph_prompt_params = [param for name, param in model.named_parameters() if name in graph_prompt_param_names]

    t1_param_ids = set()
    if not FREEZE_T1:
        for group in t1_param_groups:
            for p in group['params']:
                t1_param_ids.add(id(p))
    other_trainable_params = []
    for name, param in model.named_parameters():
        if name in graph_prompt_param_names:
            continue
        if id(param) in t1_param_ids:
            continue
        if param.requires_grad:
            other_trainable_params.append(param)

    graph_prompt_group = {'params': graph_prompt_params, 'lr': EXPERT_LR}
    other_group = {'params': other_trainable_params, 'lr': EXPERT_LR}
    all_param_groups = []
    if not FREEZE_T1:
        all_param_groups.extend(t1_param_groups)
    all_param_groups.extend([graph_prompt_group, other_group])
    optimizer = torch.optim.AdamW(all_param_groups, weight_decay=WEIGHT_DECAY)
    optimizer_graph_prompt_group = graph_prompt_group
    optimizer_other_group = other_group

    print(f"\n可训练参数统计:")
    if not FREEZE_T1:
        print(f"  T1 分层参数组数量: {len(t1_param_groups)}")
    print(f"  GraphPrompt 可训练参数数量: {sum(p.numel() for p in graph_prompt_params):,}")
    print(f"  其他可训练参数数量: {sum(p.numel() for p in other_trainable_params):,}")
    print(f"  总计可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    scaler = GradScaler('cuda', enabled=USE_AMP)
    class_weight = torch.tensor([nc_weight, pd_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    print(f"[数据平衡] 使用的类别权重: NC={nc_weight:.4f}, PD={pd_weight:.4f}")
    print(
        f"[Contrastive] enabled | weight={CONTRAST_LOSS_WEIGHT:.4f} | "
        f"temperature={CONTRAST_TEMPERATURE:.4f} | projection_dim=128"
    )

    # 训练准备
    best_score = -np.inf
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    model.rot_counts = [0] * len(model.i2moe.interaction_experts)

    print("\n开始训练...")
    for epoch in range(1, NUM_EPOCHS + 1):
        # 学习率调度
        if epoch <= WARMUP_EPOCHS:
            current_lr = EXPERT_LR * epoch / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)
            current_lr = MIN_LR + (EXPERT_LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            if pg is optimizer_graph_prompt_group or pg is optimizer_other_group:
                pg['lr'] = current_lr

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_task_loss = 0.0
        train_interaction_loss = 0.0
        train_contrast_loss = 0.0
        train_correct = 0
        train_total = 0
        for t1_imgs, fmri_mats, adjs, labels in train_loader:
            t1_imgs = t1_imgs.to(device)
            fmri_mats = fmri_mats.to(device)
            adjs = adjs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=USE_AMP):
                outputs, interaction_losses, contrast_loss = model(
                    t1_imgs, fmri_mats, adjs,
                    parc_type='aal116', disease_type='PD',
                    return_contrast_loss=True
                )
                task_loss = criterion(outputs, labels)
                interaction_loss = sum(interaction_losses) / len(interaction_losses)

                # Total training objective:
                # classification loss + I2MoE interaction loss + T1-fMRI contrastive loss.
                # contrast_loss is computed from paired embeddings before InteractionMoE.
                total_loss = (
                    task_loss
                    + model.interaction_loss_weight * interaction_loss
                    + CONTRAST_LOSS_WEIGHT * contrast_loss
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item() * t1_imgs.size(0)
            train_task_loss += task_loss.item() * t1_imgs.size(0)
            train_interaction_loss += interaction_loss.item() * t1_imgs.size(0)
            train_contrast_loss += contrast_loss.item() * t1_imgs.size(0)
            pred = outputs.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_task_loss = train_task_loss / train_total
        epoch_interaction_loss = train_interaction_loss / train_total
        epoch_contrast_loss = train_contrast_loss / train_total
        epoch_weighted_contrast_loss = CONTRAST_LOSS_WEIGHT * epoch_contrast_loss
        epoch_train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for t1_imgs, fmri_mats, adjs, labels in val_loader:
                t1_imgs = t1_imgs.to(device)
                fmri_mats = fmri_mats.to(device)
                adjs = adjs.to(device)
                labels = labels.to(device)
                with autocast(device_type='cuda', enabled=USE_AMP):
                    outputs = model(t1_imgs, fmri_mats, adjs, parc_type='aal116', disease_type='PD')
                probs = torch.softmax(outputs.float(), dim=1)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_f1 = f1_score(val_labels, val_preds, average='binary')

        # 计算验证集混淆矩阵
        # 标签约定: 0 = NC, 1 = PD
        tn, fp, fn, tp = confusion_matrix(
            val_labels,
            val_preds,
            labels=[0, 1]
        ).ravel()

        # Sensitivity / Recall: PD 样本中被正确预测为 PD 的比例
        val_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity: NC 样本中被正确预测为 NC 的比例
        val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Select the checkpoint using validation AUC and binary F1 (PD positive).
        current_score = val_auc + val_f1

        print(
            f"Epoch {epoch:3d} | LR: {current_lr:.2e} | "
            f"Loss total={epoch_train_loss:.4f}, task={epoch_task_loss:.4f}, "
            f"interaction={epoch_interaction_loss:.4f}, contrast={epoch_contrast_loss:.4f}, "
            f"wContrast={epoch_weighted_contrast_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | "
            f"Val SEN: {val_sensitivity:.4f} | Val SPE: {val_specificity:.4f} | Score ({SELECTION_SCORE}): {current_score:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  保存最佳模型 (Score [{SELECTION_SCORE}]: {current_score:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  早停触发")
                break

        # RollingQ 检查（每10个epoch）
        if epoch % 10 == 0:
            print(f"\n[RollingQ] 检查 epoch {epoch} 的注意力平衡...")
            for idx, expert_wrapper in enumerate(model.i2moe.interaction_experts):
                inner_expert = expert_wrapper.fusion_model
                expert_type = 'uni' if idx < 2 else ('syn' if idx == 2 else 'red')
                if expert_type == 'red':
                    continue

                all_q = []
                all_k_t1 = []
                all_k_fmri = []
                inner_expert.eval()
                with torch.no_grad():
                    num_batches = min(20, len(train_loader))
                    train_iter = iter(train_loader)
                    for _ in range(len(train_loader) - num_batches):
                        next(train_iter)
                    for _ in range(num_batches):
                        t1_imgs, fmri_mats, adjs, _ = next(train_iter)
                        t1_imgs = t1_imgs.to(device)
                        fmri_mats = fmri_mats.to(device)
                        adjs = adjs.to(device)
                        t1_feat = model.t1_extractor(t1_imgs)
                        fmri_feat = model.fmri_extractor(fmri_mats, adjs, 'aal116', 'PD')
                        t1_emb = model.proj_t1(t1_feat)
                        fmri_emb = model.proj_fmri(fmri_feat)
                        inputs = [t1_emb, fmri_emb]
                        q, k_fmri, _, k_t1 = inner_expert.get_attention_info(inputs)
                        all_q.append(q)
                        all_k_t1.append(k_t1)
                        all_k_fmri.append(k_fmri)

                q_all = torch.cat(all_q, dim=0)
                k_t1_all = torch.cat(all_k_t1, dim=0)
                k_fmri_all = torch.cat(all_k_fmri, dim=0)

                cos_q_kT1 = torch.cosine_similarity(q_all, k_t1_all, dim=-1)
                cos_q_kF = torch.cosine_similarity(q_all, k_fmri_all, dim=-1)
                air_syn = (cos_q_kT1 - cos_q_kF).mean().item()

                if expert_type == 'uni':
                    if idx == 0:  # T1 专家
                        current_air = air_syn
                        print(f"  Expert 0 (T1专家) | AIR_T1 = {current_air:.4f}")
                        desired_air = 0.5
                        if current_air < desired_air and model.rot_counts[idx] < MAX_ROTATIONS:
                            avg_q = q_all.mean(dim=0)
                            norm_k_t1 = k_t1_all.mean(dim=0) / (k_t1_all.mean(dim=0).norm() + 1e-8)
                            target = norm_k_t1
                            target = target * (avg_q.norm() / target.norm())
                            R_full = compute_rotation_matrix(avg_q, target).to(device)
                            gap = desired_air - current_air
                            step = min(0.1, gap / desired_air * 0.5)
                            I = torch.eye(avg_q.size(0), device=device)
                            R_step = I + step * (R_full - I)
                            inner_expert.rot_T1 = R_step @ inner_expert.rot_T1
                            model.rot_counts[idx] += 1
                            q_rot = q_all @ R_step.T
                            new_air = (torch.cosine_similarity(q_rot, k_t1_all, dim=-1) -
                                       torch.cosine_similarity(q_rot, k_fmri_all, dim=-1)).mean().item()
                            print(f"    -> 旋转 T1 专家：AIR_T1 {current_air:.4f} → {new_air:.4f} (步长={step:.3f})")
                    else:  # fMRI 专家 (idx==1)
                        current_air = -air_syn
                        print(f"  Expert 1 (fMRI专家) | AIR_fMRI = {current_air:.4f}")
                        desired_air = 0.5
                        if current_air < desired_air and model.rot_counts[idx] < MAX_ROTATIONS:
                            avg_q = q_all.mean(dim=0)
                            norm_k_fmri = k_fmri_all.mean(dim=0) / (k_fmri_all.mean(dim=0).norm() + 1e-8)
                            target = norm_k_fmri
                            target = target * (avg_q.norm() / target.norm())
                            R_full = compute_rotation_matrix(avg_q, target).to(device)
                            gap = desired_air - current_air
                            step = min(0.1, gap / desired_air * 0.5)
                            I = torch.eye(avg_q.size(0), device=device)
                            R_step = I + step * (R_full - I)
                            inner_expert.rot_T1 = R_step @ inner_expert.rot_T1
                            model.rot_counts[idx] += 1
                            q_rot = q_all @ R_step.T
                            new_air = (torch.cosine_similarity(q_rot, k_fmri_all, dim=-1) -
                                       torch.cosine_similarity(q_rot, k_t1_all, dim=-1)).mean().item()
                            print(f"    -> 旋转 fMRI 专家：AIR_fMRI {current_air:.4f} → {new_air:.4f} (步长={step:.3f})")
                elif expert_type == 'syn':
                    print(f"  Expert 2 (协同专家) | AIR_SYN = {air_syn:.4f}")
                    if abs(air_syn) > BETA_SYN and model.rot_counts[idx] < MAX_ROTATIONS:
                        alpha = 0.5 * (1 + math.tanh(-RHO * air_syn))
                        avg_q = q_all.mean(dim=0)
                        norm_k_t1 = k_t1_all.mean(dim=0) / (k_t1_all.mean(dim=0).norm() + 1e-8)
                        norm_k_fmri = k_fmri_all.mean(dim=0) / (k_fmri_all.mean(dim=0).norm() + 1e-8)
                        target = alpha * norm_k_fmri + (1 - alpha) * norm_k_t1
                        target = target * (avg_q.norm() / target.norm())
                        R = compute_rotation_matrix(avg_q, target).to(device)
                        inner_expert.rot_T1 = R @ inner_expert.rot_T1
                        model.rot_counts[idx] += 1
                        q_rot = q_all @ R.T
                        new_air = (torch.cosine_similarity(q_rot, k_t1_all, dim=-1) -
                                   torch.cosine_similarity(q_rot, k_fmri_all, dim=-1)).mean().item()
                        print(f"    -> 旋转协同专家：AIR_SYN {air_syn:.4f} → {new_air:.4f} (alpha={alpha:.3f})")

    # ======================== 测试最佳模型 ========================
    model.load_state_dict(best_model_state)
    model.eval()

    test_preds, test_labels, test_probs, test_ids = [], [], [], []
    with torch.no_grad():
        for t1_imgs, fmri_mats, adjs, labels, ids in test_loader:
            t1_imgs = t1_imgs.to(device)
            fmri_mats = fmri_mats.to(device)
            adjs = adjs.to(device)
            with autocast(device_type='cuda', enabled=USE_AMP):
                outputs = model(t1_imgs, fmri_mats, adjs, parc_type='aal116', disease_type='PD')
            probs = torch.softmax(outputs.float(), dim=1)
            preds = outputs.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())
            test_ids.extend(ids)

    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_f1 = f1_score(test_labels, test_preds, average='binary')
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)

    print(f"\n测试集结果 (split_seed={data_split_seed}, exp_seed={exp_seed}, weight={interaction_loss_weight}):")
    print(f"  Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}, Recall/Sensitivity: {test_recall:.4f}")

    # 保存错误样本
    error_ids = []
    for i, (true, pred) in enumerate(zip(test_labels, test_preds)):
        if true != pred:
            error_ids.append([test_ids[i], true, pred])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_csv_filename = f"test_errors-splitseed{data_split_seed}-expseed{exp_seed}-weight{interaction_loss_weight}-{timestamp}.csv"
    error_csv_path = os.path.join('divide', error_csv_filename)
    os.makedirs('test_error', exist_ok=True)
    with open(error_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject_id', 'true_label', 'pred_label'])
        writer.writerows(error_ids)
    print(f"测试错误样本已保存至 {error_csv_path}")

    # 混淆矩阵与模型保存
    cm = confusion_matrix(test_labels, test_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print(f"  Specificity: {specificity:.4f}")
        print(f"手动 sensitivity: {sensitivity:.4f}, sklearn recall: {recall_score(test_labels, test_preds, pos_label=1):.4f}")
        print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        model_filename = f"best_model_splitseed{data_split_seed}_expseed{exp_seed}_weight{interaction_loss_weight}.pth"
        torch.save(best_model_state, model_filename)
        print(f"最佳模型已保存至 {model_filename}")


    else:
        print("混淆矩阵:", cm)

    print(f"\n实验完成: DATA_SPLIT_SEED={data_split_seed}, EXP_SEED={exp_seed}, WEIGHT={interaction_loss_weight}\n")



    result = {
        "accuracy": float(test_acc), "auc": float(test_auc),
        "f1": float(test_f1), "precision": float(test_precision),
        "sensitivity": float(test_recall),
        "split_seed": int(data_split_seed), "training_seed": int(exp_seed),
        "interaction_loss_weight": float(interaction_loss_weight),
        "selection_score": SELECTION_SCORE,
    }
    if cm.shape == (2, 2):
        result["specificity"] = float(specificity)
        result["confusion_matrix"] = cm.tolist()
    import json
    with open("metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
