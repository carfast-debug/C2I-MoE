import torch
from torch import nn
import math


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=3,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="relu",
                 hidden_dropout_prob=0.3,
                 attention_probs_dropout_prob=0.3,
                 max_position_embeddings=512,
                 add_abs_pos_emb=False,
                 add_pos_enc=False,
                 unified_self_att=False):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.add_abs_pos_emb = add_abs_pos_emb
        self.add_pos_enc = add_pos_enc
        self.unified_self_att = unified_self_att


BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.add_abs_pos_emb = config.add_abs_pos_emb
        if self.add_abs_pos_emb:
            self.abs_pos_emb = nn.Parameter(torch.randn(512, self.attention_head_size))
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.add_abs_pos_emb:
            pos_emb = self.abs_pos_emb[0:context.size(1), :]
            pos_emb_q = self.abs_pos_emb[0:hidden_states.size(1), :]
            pos_emb_q = pos_emb_q.expand(query_layer.size(0), query_layer.size(1), -1, -1)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.add_abs_pos_emb:
            attention_pos_scores = torch.matmul(query_layer + pos_emb_q, pos_emb.transpose(-1, -2))
            attention_scores = (attention_scores + attention_pos_scores) / math.sqrt(self.attention_head_size)
        else:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand((-1, attention_scores.size(1), attention_scores.size(2), -1))
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, rotation_matrix=None):
        # 忽略 rotation_matrix，保持原有行为
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask, rotation_matrix=None):
        # 忽略 rotation_matrix
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# ======================================================================
# 带门控的注意力模块 (Gated Attention)
# ======================================================================

class GatedBertAttention(nn.Module):
    """
    带门控的多头注意力，在 SDPA 输出后乘上 sigmoid 门控。
    门控由 query 的隐藏状态（hidden_states）产生，headwise 形式：每个头一个标量。
    支持可选的旋转矩阵 (rotation_matrix) 应用于 query。
    """

    def __init__(self, config, gate_headwise=True):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.add_abs_pos_emb = config.add_abs_pos_emb
        if self.add_abs_pos_emb:
            self.abs_pos_emb = nn.Parameter(torch.randn(512, self.attention_head_size))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 门控投影
        self.gate_headwise = gate_headwise
        if self.gate_headwise:
            # 每个头一个标量：输出维度 = num_heads
            self.gate_proj = nn.Linear(config.hidden_size, self.num_attention_heads)
        else:
            # 逐元素：输出维度 = all_head_size
            self.gate_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.gate_act = nn.Sigmoid()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None, rotation_matrix=None):
        # hidden_states: Query 的来源（当前模态或自身序列）
        mixed_query_layer = self.query(hidden_states)  # [B, Sq, all_head_size]
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        # ===== RollingQ: 应用旋转矩阵到第一个 token 的查询 =====
        if rotation_matrix is not None:
            # 只旋转第一个 token 的查询 (位置0)
            q_first = mixed_query_layer[:, 0:1, :]  # [B, 1, all_head_size]
            q_first_rot = q_first @ rotation_matrix.T
            mixed_query_layer[:, 0:1, :] = q_first_rot
        # =======================================================

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [B, H, Sq, D]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # ... 其余代码保持不变，删除原来分头后的旋转代码 ...


        if self.add_abs_pos_emb:
            pos_emb = self.abs_pos_emb[0:context.size(1), :]
            pos_emb_q = self.abs_pos_emb[0:hidden_states.size(1), :]
            pos_emb_q = pos_emb_q.expand(query_layer.size(0), query_layer.size(1), -1, -1)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.add_abs_pos_emb:
            attention_pos_scores = torch.matmul(query_layer + pos_emb_q, pos_emb.transpose(-1, -2))
            attention_scores = (attention_scores + attention_pos_scores) / math.sqrt(self.attention_head_size)
        else:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand((-1, attention_scores.size(1), attention_scores.size(2), -1))
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # (B, H, Sq, D)

        # ---------- 门控注入 ----------
        gate = self.gate_proj(hidden_states)  # (B, Sq, H) or (B, Sq, H*D)
        gate = self.gate_act(gate)
        if self.gate_headwise:
            # 变形为 (B, H, Sq, 1) 以广播到 (B, H, Sq, D)
            gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (B, H, Sq, 1)
        else:
            # elementwise: 变形为 (B, Sq, H, D) 再调整为 (B, H, Sq, D)
            gate = gate.view(gate.size(0), gate.size(1), self.num_attention_heads, self.attention_head_size)
            gate = gate.permute(0, 2, 1, 3)
        context_layer = context_layer * gate
        # ---------------------------------

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer


class GatedBertCrossattLayer(nn.Module):
    def __init__(self, config, gate_headwise=True):
        super().__init__()
        self.att = GatedBertAttention(config, gate_headwise=gate_headwise)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, rotation_matrix=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask, rotation_matrix=rotation_matrix)
        attention_output = self.output(output, input_tensor)
        return attention_output


class GatedBertSelfattLayer(nn.Module):
    def __init__(self, config, gate_headwise=True):
        super().__init__()
        self.self = GatedBertAttention(config, gate_headwise=gate_headwise)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask, rotation_matrix=None):
        self_output = self.self(input_tensor, input_tensor, attention_mask, rotation_matrix=rotation_matrix)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


# ======================================================================
# 交叉模态编码器层（可选择是否使用门控）
# ======================================================================

class CMELayer(nn.Module):
    def __init__(self, config, use_gated=False):
        super().__init__()
        # 交叉注意力（不变）
        if use_gated:
            self.audio_attention = GatedBertCrossattLayer(config)
            self.lang_attention = GatedBertCrossattLayer(config)
        else:
            self.audio_attention = BertCrossattLayer(config)
            self.lang_attention = BertCrossattLayer(config)

        # 始终创建分开的自注意力层（供前几层使用）
        if use_gated:
            self.lang_self_att = GatedBertSelfattLayer(config)
            self.audio_self_att = GatedBertSelfattLayer(config)
        else:
            self.lang_self_att = BertSelfattLayer(config)
            self.audio_self_att = BertSelfattLayer(config)

        # 如果启用统一自注意力，额外创建统一层
        self.unified_self_att = config.unified_self_att
        if self.unified_self_att:
            attn_class = GatedBertAttention if use_gated else BertAttention
            self.unified_attn = attn_class(config)
            self.unified_attn_output = BertAttOutput(config)
            self.unified_cls = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            nn.init.trunc_normal_(self.unified_cls, std=0.02)

        # FFN（不变）
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.audio_inter = BertIntermediate(config)
        self.audio_output = BertOutput(config)


    def cross_att(self, lang_input, lang_attention_mask, audio_input, audio_attention_mask,
                  rot_T1=None, rot_fmri=None):
        lang_att_output = self.lang_attention(lang_input, audio_input,
                                              ctx_att_mask=audio_attention_mask,
                                              rotation_matrix=rot_T1)
        audio_att_output = self.audio_attention(audio_input, lang_input,
                                                ctx_att_mask=lang_attention_mask,
                                                rotation_matrix=rot_fmri)
        return lang_att_output, audio_att_output

    def unified_self_attention(self, lang_input, lang_mask, audio_input, audio_mask,
                               rotation_matrix=None):
        """
        统一自注意力：拼接 [unified_cls, lang_input, audio_input]
        返回拆分后的 lang_output, audio_output
        """
        B, L_lang, D = lang_input.shape
        B, L_audio, D = audio_input.shape

        # 扩展 unified_cls 到 batch
        cls_token = self.unified_cls.expand(B, -1, -1)   # [B,1,D]
        # 拼接
        combined = torch.cat([cls_token, lang_input, audio_input], dim=1)   # [B, 1+L_lang+L_audio, D]

        # 构造注意力掩码：所有有效位置为1，无padding掩码（这里假设没有padding）
        # 如果需要支持 padding，可扩展 mask。简化起见，我们使用全1 mask。
        combined_mask = torch.ones(B, combined.size(1), device=lang_input.device)

        # 通过统一注意力层（支持旋转矩阵）
        attn_output = self.unified_attn(combined, combined, combined_mask, rotation_matrix=rotation_matrix)
        attn_output = self.unified_attn_output(attn_output, combined)

        # 拆分
        cls_out = attn_output[:, 0:1, :]          # [B,1,D]（用于 RollingQ 旋转后输出）
        lang_out = attn_output[:, 1:1+L_lang, :]   # [B,L_lang,D]
        audio_out = attn_output[:, 1+L_lang:, :]   # [B,L_audio,D]

        return lang_out, audio_out

    def self_att(self, lang_input, lang_attention_mask, audio_input, audio_attention_mask,
                 rot_T1=None, rot_fmri=None, use_unified=None):
        # 如果外部传入了 use_unified，则使用它；否则使用 self.unified_self_att
        effective_unified = use_unified if use_unified is not None else self.unified_self_att
        if effective_unified:
            # 使用统一自注意力（只需要一个旋转矩阵，约定使用 rot_T1）
            return self.unified_self_attention(lang_input, lang_attention_mask,
                                               audio_input, audio_attention_mask,
                                               rotation_matrix=rot_T1)
        else:
            # 使用原来的分开自注意力
            lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
            audio_att_output = self.audio_self_att(audio_input, audio_attention_mask)
            return lang_att_output, audio_att_output

    def output_fc(self, lang_input, audio_input):
        lang_inter_output = self.lang_inter(lang_input)
        audio_inter_output = self.audio_inter(audio_input)
        lang_output = self.lang_output(lang_inter_output, lang_input)
        audio_output = self.audio_output(audio_inter_output, audio_input)
        return lang_output, audio_output

    def forward(self, lang_feats, lang_attention_mask, audio_feats, audio_attention_mask,
                rot_T1=None, rot_fmri=None, use_unified_self_att=None):

        if use_unified_self_att is None:
            use_unified_self_att = self.unified_self_att

        # 1) 交叉注意力（可能传入旋转矩阵）
        lang_att_output, audio_att_output = self.cross_att(
            lang_feats, lang_attention_mask, audio_feats, audio_attention_mask,
            rot_T1=rot_T1, rot_fmri=rot_fmri
        )
        # 2) 自注意力（支持统一模式）
        lang_att_output, audio_att_output = self.self_att(
            lang_att_output, lang_attention_mask, audio_att_output, audio_attention_mask,
            rot_T1=rot_T1, rot_fmri=rot_fmri,
            use_unified=use_unified_self_att
        )


        # 3) FFN
        lang_output, audio_output = self.output_fc(lang_att_output, audio_att_output)
        return lang_output, audio_output

    def get_unified_attention_info(self, lang_input, lang_mask, audio_input, audio_mask):
        """
        专门用于获取统一自注意力中的 [CLS] 查询和两个模态的键（均值）。
        仅用于评估，不应用旋转矩阵。
        返回: (cls_query, mean_k_T1, mean_k_fmri)
        """
        if not self.unified_self_att:
            raise RuntimeError("get_unified_attention_info 只能在 unified_self_att=True 时调用")

        B, L_lang, D = lang_input.shape
        B, L_audio, D = audio_input.shape

        # 扩展统一 CLS
        cls_token = self.unified_cls.expand(B, -1, -1)  # [B,1,D]
        combined = torch.cat([cls_token, lang_input, audio_input], dim=1)  # [B, 1+L_lang+L_audio, D]

        # 直接通过 attention 层获取查询和键（不应用旋转矩阵，因为 rotation_matrix=None）
        # 注意这里我们只使用 unifed_attn 内部的 query/key 投影，不经过整个 attention 计算
        # 为了获取未旋转的查询，我们直接调用 query 线性层（因为 rotation_matrix 为 None 时不会旋转）
        q_all = self.unified_attn.query(combined)  # [B, 1+L_lang+L_audio, H*D]
        k_all = self.unified_attn.key(combined)  # [B, 1+L_lang+L_audio, H*D]

        # 取 CLS token 的查询（位置 0）
        cls_query = q_all[:, 0, :]  # [B, H*D]

        # 分别取 T1 和 fMRI 部分的所有 token 的键（包括各自的 CLS token？原设计中 lang_input 已经包含了它自己的 CLS）
        # lang_input 的第一个 token 是 T1 的 cls，后面是特征 token；音频同理。
        # 我们取所有 token 的键的平均值作为该模态的“平均键”
        k_T1 = k_all[:, 1:1 + L_lang, :]  # [B, L_lang, H*D]
        k_fmri = k_all[:, 1 + L_lang:, :]  # [B, L_audio, H*D]

        # 对每个样本，计算该模态所有 token 键的均值
        mean_k_T1 = k_T1.mean(dim=1)  # [B, H*D]
        mean_k_fmri = k_fmri.mean(dim=1)  # [B, H*D]

        return cls_query, mean_k_T1, mean_k_fmri
