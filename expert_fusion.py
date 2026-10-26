import torch
import torch.nn as nn
from attention.cross_attn_encoder import CMELayer, BertConfig

class GatedAttentionExpert(nn.Module):
    """基于门控交叉注意力的 I²MoE 专家，支持 RollingQ 旋转矩阵"""
    def __init__(self,
                 hidden_size=768,
                 num_layers=3,
                 num_classes=2,
                 dropout=0.3,
                 gated_layers=[2]):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.t1_cls_emb = nn.Embedding(1, hidden_size)
        self.fmri_cls_emb = nn.Embedding(1, hidden_size)

        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            unified_self_att=True  # 启用统一自注意力
        )

        self.cme_layers = nn.ModuleList()
        for i in range(num_layers):
            use_gated = (i in gated_layers)
            self.cme_layers.append(CMELayer(config, use_gated=use_gated))

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

        # 注册旋转矩阵（单位矩阵，不参与训练）
        self.register_buffer('rot_T1', torch.eye(hidden_size))
        self.register_buffer('rot_fmri', torch.eye(hidden_size))

    def forward(self, inputs):
        t1_feat, fmri_feat = inputs[0], inputs[1]
        batch_size = t1_feat.shape[0]
        device = t1_feat.device

        t1_seq = t1_feat.unsqueeze(1)
        fmri_seq = fmri_feat.unsqueeze(1)

        t1_cls = self.t1_cls_emb(torch.zeros(batch_size, dtype=torch.long, device=device)).unsqueeze(1)
        fmri_cls = self.fmri_cls_emb(torch.zeros(batch_size, dtype=torch.long, device=device)).unsqueeze(1)

        t1_seq = torch.cat([t1_cls, t1_seq], dim=1)
        fmri_seq = torch.cat([fmri_cls, fmri_seq], dim=1)

        t1_mask = torch.ones(batch_size, 2, device=device)
        fmri_mask = torch.ones(batch_size, 2, device=device)

        # 前 N-1 层不加旋转，且使用分开自注意力（use_unified_self_att=False）
        for i in range(self.num_layers - 1):
            t1_seq, fmri_seq = self.cme_layers[i](t1_seq, t1_mask, fmri_seq, fmri_mask,
                                                  use_unified_self_att=False)

        # 最后一层使用统一自注意力，并传入旋转矩阵
        last_layer = self.cme_layers[-1]
        t1_seq, fmri_seq = last_layer(t1_seq, t1_mask, fmri_seq, fmri_mask,
                                          rot_T1=self.rot_T1, rot_fmri=self.rot_fmri,
                                          use_unified_self_att=True)

        fused = torch.cat([t1_seq[:, 0, :], fmri_seq[:, 0, :]], dim=-1)
        logits = self.classifier(fused)
        return logits

    def get_attention_info(self, inputs):
        """
        返回用于 RollingQ 评估的信息。
        对于统一自注意力模式，返回 (cls_query, mean_k_fmri, cls_query, mean_k_T1)
        以兼容训练代码中的 (q_T1, k_fmri, q_fmri, k_t1) 接口。
        """
        t1_feat, fmri_feat = inputs[0], inputs[1]
        batch_size = t1_feat.shape[0]
        device = t1_feat.device

        t1_seq = t1_feat.unsqueeze(1)
        fmri_seq = fmri_feat.unsqueeze(1)

        t1_cls = self.t1_cls_emb(torch.zeros(batch_size, dtype=torch.long, device=device)).unsqueeze(1)
        fmri_cls = self.fmri_cls_emb(torch.zeros(batch_size, dtype=torch.long, device=device)).unsqueeze(1)

        t1_seq = torch.cat([t1_cls, t1_seq], dim=1)   # [B, 2, H]
        fmri_seq = torch.cat([fmri_cls, fmri_seq], dim=1)

        t1_mask = torch.ones(batch_size, 2, device=device)
        fmri_mask = torch.ones(batch_size, 2, device=device)

        # 前 N-1 层（不使用统一自注意力）
        for i in range(self.num_layers - 1):
            t1_seq, fmri_seq = self.cme_layers[i](t1_seq, t1_mask, fmri_seq, fmri_mask,
                                                  use_unified_self_att=False)

        # 检查最后一层是否使用了统一自注意力
        last_layer = self.cme_layers[-1]
        if last_layer.unified_self_att:
            # 使用统一自注意力模式获取信息
            cls_query, mean_k_T1, mean_k_fmri = last_layer.get_unified_attention_info(
                t1_seq, t1_mask, fmri_seq, fmri_mask
            )
            # 为了兼容现有训练代码，我们将 q_T1 和 q_fmri 都设为 cls_query
            return cls_query, mean_k_fmri, cls_query, mean_k_T1
        else:
            # 旧的分开自注意力模式（保留原逻辑）
            lang_att = last_layer.lang_attention
            audio_att = last_layer.audio_attention
            q_t1_all = lang_att.att.query(t1_seq)
            k_fmri_all = lang_att.att.key(fmri_seq)
            q_fmri_all = audio_att.att.query(fmri_seq)
            k_t1_all = audio_att.att.key(t1_seq)
            q_t1 = q_t1_all[:, 0, :]
            q_fmri = q_fmri_all[:, 0, :]
            k_fmri = k_fmri_all
            k_t1 = k_t1_all
            return q_t1, k_fmri, q_fmri, k_t1
