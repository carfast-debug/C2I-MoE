import torch
import torch.nn as nn
import torch.nn.functional as F


from encoders.model_3dvit import vit_3dino_large
from encoders.brain_gfm import BrainGFM
from attention.cross_attn_encoder import CMELayer, BertConfig

from InteractionMoE import InteractionMoE          # 从 I²MoE 源码复制并微调
#
from expert_fusion import GatedAttentionExpert



def load_3dino_weights(model, weight_path):
    state = torch.load(weight_path, map_location='cpu', weights_only=True)
    teacher = state['teacher']
    new_state_dict = {}
    for key, value in teacher.items():
        if key.startswith('backbone.'):
            new_key = key[9:]
            new_state_dict[new_key] = value
    for k in list(new_state_dict.keys()):
        if k.startswith('dino_head.') or k.startswith('ibot_head.') or k.startswith('head.'):
            del new_state_dict[k]
    model.load_state_dict(new_state_dict, strict=False)
    return model


class T1FeatureExtractor(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.backbone = vit_3dino_large(volume_size=112, patch_size=16,
                                        in_channels=1, num_classes=2)
        if pretrained_path:
            load_3dino_weights(self.backbone, pretrained_path)
        self.backbone.head = nn.Identity()

    def forward(self, x):
        feat = self.backbone.forward_features(x)   # (B, 1024)
        return feat


class FMRIFeatureExtractor(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.backbone = BrainGFM(
            ff_hidden_size=256, num_classes=2, num_self_att_layers=4,
            dropout=0.3, num_GNN_layers=4, nhead=8, hidden_dim=256,   # 改为256
            max_feature_dim=512, rwse_steps=5, max_nodes=512, moe_num_experts=1
        )
        # ... 其余不变
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    new_k = k[8:]
                    new_state_dict[new_k] = v
            # 过滤并加载
            model_dict = self.backbone.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}
            print(f"成功匹配的预训练权重数量: {len(pretrained_dict)} / {len(model_dict)}")
            print("匹配的键示例（前5个）:", list(pretrained_dict.keys())[:5])
            self.backbone.load_state_dict(pretrained_dict, strict=False)

    def forward(self, node_feat, adj, parc_type, disease_type, valid_num_nodes=None):
        g = self.backbone(node_feat, adj, parc_type, disease_type, valid_num_nodes)
        return g   # (B, 256)

    def forward_with_attention(self, node_feat, adj, parc_type, disease_type, valid_num_nodes=None):
        return self.backbone.forward_with_attention(
            node_feat, adj, parc_type, disease_type, valid_num_nodes
        )


class T1_FMRI_CME(nn.Module):
    def __init__(self, t1_extractor, fmri_extractor, hidden_size=768, num_layers=3, num_classes=2,
                 dropout=0.3, gated_layers=None):
        """
        gated_layers: list of int, 指定哪些层使用门控注意力（例如 [2] 表示最后一层）。
                      默认为 None，即所有层均不使用门控。
        """
        super().__init__()
        self.t1_extractor = t1_extractor
        self.fmri_extractor = fmri_extractor

        self.t1_proj = nn.Linear(1024, hidden_size)
        self.fmri_proj = nn.Linear(256, hidden_size)

        self.t1_cls_emb = nn.Embedding(1, hidden_size)
        self.fmri_cls_emb = nn.Embedding(1, hidden_size)

        config = BertConfig(hidden_size=hidden_size,
                            num_hidden_layers=num_layers,
                            hidden_dropout_prob=dropout,
                            attention_probs_dropout_prob=dropout)

        if gated_layers is None:
            gated_layers = []

        self.cme_layers = nn.ModuleList()
        for i in range(num_layers):
            use_gated = (i in gated_layers)
            self.cme_layers.append(CMELayer(config, use_gated=use_gated))

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, t1_imgs, fmri_node_feat, fmri_adj,
                parc_type='aal116', disease_type='PD', valid_num_nodes=None,
                return_attn_info=False):
        t1_feat = self.t1_extractor(t1_imgs)
        fmri_feat = self.fmri_extractor(fmri_node_feat, fmri_adj,
                                        parc_type, disease_type, valid_num_nodes)
        if return_attn_info:
            out, attn_info = self.fuse_features(t1_feat, fmri_feat, return_attn_info=True)
            return out, attn_info
        else:
            return self.fuse_features(t1_feat, fmri_feat)

    def fuse_features(self, t1_feat, fmri_feat, return_attn_info=False):
        t1_seq = self.t1_proj(t1_feat).unsqueeze(1)
        fmri_seq = self.fmri_proj(fmri_feat).unsqueeze(1)

        batch_size = t1_seq.size(0)
        device = t1_seq.device
        t1_cls = self.t1_cls_emb(torch.zeros(batch_size, dtype=torch.long, device=device)).unsqueeze(1)
        fmri_cls = self.fmri_cls_emb(torch.zeros(batch_size, dtype=torch.long, device=device)).unsqueeze(1)
        t1_seq = torch.cat([t1_cls, t1_seq], dim=1)
        fmri_seq = torch.cat([fmri_cls, fmri_seq], dim=1)

        t1_mask = torch.ones(batch_size, 2, device=device)
        fmri_mask = torch.ones(batch_size, 2, device=device)

        t1_input_last = None
        fmri_input_last = None
        for i, layer in enumerate(self.cme_layers):
            if i == len(self.cme_layers) - 1 and return_attn_info:
                t1_input_last = t1_seq.clone()
                fmri_input_last = fmri_seq.clone()
            t1_seq, fmri_seq = layer(t1_seq, t1_mask, fmri_seq, fmri_mask)

        fused = torch.cat([t1_seq[:, 0, :], fmri_seq[:, 0, :]], dim=-1)
        out = self.classifier(fused)

        if return_attn_info:
            with torch.no_grad():
                last_layer = self.cme_layers[-1]
                lang_att = last_layer.lang_attention
                audio_att = last_layer.audio_attention

                q_t1_all = lang_att.att.query(t1_input_last)
                k_fmri_all = lang_att.att.key(fmri_input_last)
                q_fmri_all = audio_att.att.query(fmri_input_last)
                k_t1_all = audio_att.att.key(t1_input_last)

                t1_sa = last_layer.lang_self_att
                fmri_sa = last_layer.audio_self_att
                q_t1_self_all = t1_sa.self.query(t1_input_last)
                k_t1_self_all = t1_sa.self.key(t1_input_last)
                q_fmri_self_all = fmri_sa.self.query(fmri_input_last)
                k_fmri_self_all = fmri_sa.self.key(fmri_input_last)

                attn_info = {
                    'q_t1': q_t1_all[:, 0, :],
                    'k_fmri': k_fmri_all,
                    'q_fmri': q_fmri_all[:, 0, :],
                    'k_t1': k_t1_all,
                    'q_t1_self': q_t1_self_all[:, 0, :],
                    'k_t1_self': k_t1_self_all,
                    'q_fmri_self': q_fmri_self_all[:, 0, :],
                    'k_fmri_self': k_fmri_self_all,
                }
            return out, attn_info
        return out


# fusion_model.py 中的 I2MoE_Fusion
class I2MoE_Fusion(nn.Module):
    def __init__(self, t1_extractor, fmri_extractor,
                 hidden_size=768,
                 num_classes=2, dropout=0.3,
                 interaction_loss_weight=0.5, temperature=2,
                 num_layers=3, gated_layers=[2],
                 contrast_dim=128, contrast_temperature=0.07):
        super().__init__()
        self.t1_extractor = t1_extractor
        self.fmri_extractor = fmri_extractor

        # 外部投影层，将原始特征映射到统一维度 hidden_size
        self.proj_t1 = nn.Linear(1024, hidden_size)
        self.proj_fmri = nn.Linear(256, hidden_size)

        # Cross-modal contrastive learning branch.
        # PairedDataset returns aligned T1/fMRI samples, so the i-th T1 embedding
        # and the i-th fMRI embedding in a mini-batch form a positive pair.
        self.contrast_temperature = contrast_temperature
        self.t1_contrast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, contrast_dim)
        )
        self.fmri_contrast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, contrast_dim)
        )

        def expert_factory():
            return GatedAttentionExpert(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                gated_layers=gated_layers
            )

        self.i2moe = InteractionMoE(
            num_modalities=2,
            fusion_model_factory=expert_factory,
            fusion_sparse=False,
            hidden_dim=hidden_size,
            hidden_dim_rw=256,
            num_layer_rw=2,
            temperature_rw=temperature
        )
        self.interaction_loss_weight = interaction_loss_weight

    def compute_contrastive_loss(self, t1_emb, fmri_emb):
        """Compute bidirectional InfoNCE loss between T1 and fMRI embeddings.

        The diagonal entries of the similarity matrix are positive pairs, and
        off-diagonal entries are in-batch negatives.
        """
        z_t1 = F.normalize(self.t1_contrast_head(t1_emb), dim=-1)
        z_fmri = F.normalize(self.fmri_contrast_head(fmri_emb), dim=-1)

        logits = torch.matmul(z_t1, z_fmri.t()) / self.contrast_temperature
        targets = torch.arange(z_t1.size(0), device=z_t1.device)

        loss_t1_to_fmri = F.cross_entropy(logits, targets)
        loss_fmri_to_t1 = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_t1_to_fmri + loss_fmri_to_t1)

    def forward(self, t1_imgs, fmri_mats, adjs, parc_type='aal116', disease_type='PD',
        return_contrast_loss=False):
        t1_feat = self.t1_extractor(t1_imgs)  # (B, 1024)
        fmri_feat = self.fmri_extractor(fmri_mats, adjs, parc_type, disease_type)  # (B, 256)

        # These shared embeddings are used by both InteractionMoE and the
        # auxiliary T1-fMRI contrastive alignment objective.
        t1_emb = self.proj_t1(t1_feat)  # (B, hidden_size)
        fmri_emb = self.proj_fmri(fmri_feat)  # (B, hidden_size)
        inputs = [t1_emb, fmri_emb]
        if self.training:
            _, _, weighted_logits, interaction_losses = self.i2moe(inputs)
            if return_contrast_loss:
                # Only training requests this auxiliary loss. Validation and test
                # keep the original logits-only inference interface.
                contrast_loss = self.compute_contrastive_loss(t1_emb, fmri_emb)
                return weighted_logits, interaction_losses, contrast_loss
            return weighted_logits, interaction_losses
        else:
            _, _, weighted_logits = self.i2moe.inference(inputs)
            return weighted_logits
