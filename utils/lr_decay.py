import json
import torch
import torch.nn as nn


def _get_vit_blocks(model):
    if hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
        return model.encoder.blocks
    if hasattr(model, "blocks"):
        return model.blocks
    raise AttributeError("model must have either model.encoder.blocks or model.blocks")


def _count_real_blocks(blocks):
    count = 0
    for blk in blocks:
        if isinstance(blk, (nn.ModuleList, nn.Sequential)):
            for subblk in blk:
                if not isinstance(subblk, nn.Identity):
                    count += 1
        else:
            if not isinstance(blk, nn.Identity):
                count += 1
    return count


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    param_group_names = {}
    param_groups = {}

    blocks = _get_vit_blocks(model)
    num_real_blocks = _count_real_blocks(blocks)
    num_layers = num_real_blocks + 1

    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    if name in ["cls_token", "pos_embed", "mask_token"]:
        return 0

    if name.startswith("patch_embed"):
        return 0

    if name.startswith("blocks"):
        parts = name.split(".")
        try:
            # Supports both normal ViT: blocks.0.xxx
            # and 3DINO BlockChunk: blocks.1.6.xxx
            if len(parts) > 2 and parts[1].isdigit() and parts[2].isdigit():
                global_block_idx = int(parts[2])
                return min(global_block_idx + 1, num_layers - 1)

            if len(parts) > 1 and parts[1].isdigit():
                block_idx = int(parts[1])
                return min(block_idx + 1, num_layers - 1)
        except ValueError:
            pass

    return num_layers
