# braingfm.py
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
from .disease_names import disease_names
import os
# Import GraphPrompt module
from .graph_prompt import GraphPrompt

# ===========================================
# Disease embeddings extracted by ClinicalBERT (with fallback)
# ===========================================
def get_disease_embeddings(embed_dim=768, seed=42):
    try:
        from transformers import AutoTokenizer, AutoModel
        # 获取本文件所在目录，Bio_ClinicalBERT 放在同目录下
        local_path = os.environ.get("C2IMOE_CLINICALBERT_DIR", os.path.join(os.path.dirname(__file__), "Bio_ClinicalBERT"))
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        model = AutoModel.from_pretrained(local_path, local_files_only=True)
        model.eval()
        embeddings = {}
        with torch.no_grad():
            for disease, name in disease_names.items():
                inputs = tokenizer(name, return_tensors='pt')
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings[disease.lower()] = cls_embedding.squeeze(0).cpu()
        return embeddings
    except Exception:
        g = torch.Generator().manual_seed(seed)
        embeddings = {}
        for disease in disease_names.keys():
            embeddings[disease.lower()] = torch.randn(embed_dim, generator=g)
        return embeddings


# ===========================================
# FastMoE modules: FFN and GCN
# ===========================================
class FastMoEFFN(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, H]
        """
        B, N, H = x.shape
        scores = self.router(x.mean(dim=1))  # [B, E]
        top1 = torch.argmax(scores, dim=-1)  # [B]
        out = torch.zeros_like(x)
        for i in range(self.num_experts):
            idx = (top1 == i)
            if idx.sum() == 0:
                continue
            out[idx] = self.experts[i](x[idx])
        return out


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.act = act
        self.bn = nn.BatchNorm1d(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, adj):
        """
        Args:
            x:   [B, N, Fin]
            adj: [B, N, N]
        """
        support = torch.bmm(
            x,
            self.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        )  # [B, N, Fout]
        out = torch.bmm(adj, support)  # [B, N, Fout]
        if self.bias is not None:
            out = out + self.bias
        out = self.bn(out.view(-1, out.shape[-1])).view(out.shape)
        return self.act(out)


class FastMoEGCN(nn.Module):
    def __init__(self, hidden_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x, adj):
        """
        Args:
            x:   [B, N, H]
            adj: [B, N, N]
        """
        B, N, H = x.shape
        scores = self.router(x.mean(dim=1))  # [B, E]
        top1 = torch.argmax(scores, dim=-1)  # [B]
        out = torch.zeros_like(x)
        for i in range(self.num_experts):
            idx = (top1 == i)
            if idx.sum() == 0:
                continue
            out[idx] = self.experts[i](x[idx], adj[idx])
        return out


# ===========================================
# UGFormer encoder layer
# ===========================================
class GTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, num_experts=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FastMoEFFN(d_model, dim_feedforward, num_experts, dropout)

    def forward(self, src, src_mask=None, is_causal=None, src_key_padding_mask=None, return_attention=False):
        attn_out, attn_weights = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            need_weights=return_attention,
            average_attn_weights=False
        )
        src = self.norm1(src + self.dropout(attn_out))
        ff_out = self.ffn(src)
        src = self.norm2(src + self.dropout(ff_out))
        if return_attention:
            return src, attn_weights
        return src


# ===========================================
# BrainGFM encoder (with GraphPrompt)
# ===========================================
class BrainGFM(nn.Module):
    def __init__(self, ff_hidden_size, num_classes, num_self_att_layers, dropout, num_GNN_layers, nhead,
                 hidden_dim=128, max_feature_dim=256, rwse_steps=5, max_nodes=512, moe_num_experts=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_feature_dim = max_feature_dim
        self.rwse_steps = rwse_steps
        self.max_nodes = max_nodes

        # Project input features to hidden_dim
        self.projection_layer = nn.Linear(self.max_feature_dim, self.hidden_dim)
        self.disease_proj = nn.Linear(768, self.hidden_dim)

        # Parcellation tokens (stored in original feature space; projected later)
        self.parcellation_tokens = nn.ParameterDict({
            'schaefer':    nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'schaefer200': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'schaefer300': nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'shen268':     nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'power264':    nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'gordon333':   nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'aal116':      nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
            'aal3v1':      nn.Parameter(torch.randn(1, 1, self.max_feature_dim)),
        })

        # Disease tokens (registered as nn.Parameter)
        disease_embed_dict = get_disease_embeddings()
        self.disease_embeddings = {}
        for k, v in disease_embed_dict.items():
            param = nn.Parameter(v.unsqueeze(0).unsqueeze(0))  # [1,1,768]
            self.disease_embeddings[k.lower()] = param
            self.register_parameter(f'disease_embedding_{k.lower()}', param)

        # Unified GraphPrompt module
        self.graph_prompt = GraphPrompt(
            hidden_dim=hidden_dim,
            max_nodes=max_nodes,
            max_feature_dim=max_feature_dim,
            node_mode="scale",
            edge_strength=0.3,
            init_std=1e-2
        )

        # Stacked UGFormer blocks
        self.ugformer_layers = nn.ModuleList([
            TransformerEncoder(
                GTransformerEncoderLayer(
                    d_model=hidden_dim, nhead=nhead,
                    dim_feedforward=ff_hidden_size, dropout=dropout, num_experts=moe_num_experts
                ),
                num_layers=num_self_att_layers
            )
            for _ in range(num_GNN_layers)
        ])

        # Stacked MoE-GCN blocks
        self.lst_gnn = nn.ModuleList([FastMoEGCN(hidden_dim, moe_num_experts) for _ in range(num_GNN_layers)])
        self.predictions = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_GNN_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_GNN_layers)])

    def compute_rwse(self, adj, k):
        """
        A simple RWSE implementation using diagonal entries of k-step random walks.
        """
        B, N, _ = adj.shape
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        rw = adj.clone()
        diag_features = []
        for _ in range(k):
            rw_diag = torch.diagonal(rw, dim1=1, dim2=2).unsqueeze(-1)  # [B, N, 1]
            diag_features.append(rw_diag)
            rw = torch.bmm(rw, adj)
        return torch.cat(diag_features, dim=-1)  # [B, N, k]

    def expand_adj_block(self, adj, num_tokens=2):
        """
        Expand adjacency to include token nodes (disease + parcellation).
        """
        B, N, _ = adj.shape
        new_N = N + num_tokens
        new_adj = torch.zeros(B, new_N, new_N, device=adj.device)
        new_adj[:, num_tokens:, num_tokens:] = adj

        # Weakly connect tokens to all nodes (value=1 by default; can be reduced to e.g. 0.5 if needed)
        for i in range(num_tokens):
            new_adj[:, i, :] = 1
            new_adj[:, :, i] = 1

        # Remove self-loops (zero diagonal)
        new_adj = new_adj - torch.diag_embed(torch.diagonal(new_adj, dim1=1, dim2=2))
        return new_adj

    def forward(self, node_features, Adj_block, parc_type, disease_type, valid_num_nodes=None):
        """
        Args:
            node_features:   [B, N, F]
            Adj_block:       [B, N, N] (binary or weighted adjacency)
            parc_type:       str (must be a key in self.parcellation_tokens)
            disease_type:    str (lower-cased internally; fallback to 'none' if missing)
            valid_num_nodes: List[int] (number of valid nodes per sample; used for padding mask)
        """
        B, N, F = node_features.shape
        device = node_features.device
        if valid_num_nodes is None:
            valid_num_nodes = [N] * B

        # 1) Append RWSE features
        rwse = self.compute_rwse(Adj_block, k=self.rwse_steps)                 # [B, N, k]
        node_features = torch.cat([node_features, rwse], dim=-1)               # [B, N, F+k]

        # 2) Zero-pad to max_feature_dim
        padded = torch.zeros((B, N, self.max_feature_dim), device=device)      # [B, N, Dm]
        padded[:, :, :node_features.shape[-1]] = node_features

        # 3) Prepare tokens (projected)
        parc_token = self.projection_layer(self.parcellation_tokens[parc_type].expand(B, 1, -1))  # [B,1,H]
        disease_type = disease_type.lower()
        if disease_type not in self.disease_embeddings:
            disease_type = 'none'
        disease_token = self.disease_proj(self.disease_embeddings[disease_type].expand(B, 1, -1)) # [B,1,H]

        # 4) Apply GraphPrompt (before projecting node features to hidden_dim)
        x_prompted, A_tilde, attn_bias_nodes = self.graph_prompt(
            node_feats_BND=padded,               # [B, N, Dm]
            adj_BNN=Adj_block.float(),           # [B, N, N]
            disease_token_B1H=disease_token,     # [B, 1, H]
            parc_token_B1H=parc_token,           # [B, 1, H]
            valid_num_nodes=valid_num_nodes
        )

        # 5) Project to hidden_dim
        x_proj = self.projection_layer(x_prompted)                              # [B, N, H]

        # 6) Concatenate tokens & expand adjacency
        x = torch.cat([disease_token, parc_token, x_proj], dim=1)              # [B, N+2, H]
        Adj_block_with_tokens = self.expand_adj_block(A_tilde, num_tokens=2)   # [B, N+2, N+2]

        # 7) Build padding mask
        padding_mask = torch.ones(B, N + 2, device=device).bool()
        for i, n_valid in enumerate(valid_num_nodes):
            padding_mask[i, :n_valid + 2] = False

        # 8) Stacked UGFormer + FastMoEGCN
        out = 0
        for i in range(len(self.ugformer_layers)):
            # If you need a batched attn_mask, pad attn_bias_nodes to [B, L, L] and pass via src_mask=
            h = self.ugformer_layers[i](x, src_key_padding_mask=padding_mask)
            node_h = h[:, 2:, :]
            node_mask = (~padding_mask[:, 2:]).unsqueeze(-1).float()
            node_h = self.lst_gnn[i](node_h, Adj_block_with_tokens[:, 2:, 2:])
            g = (node_h * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1e-6)
            out += self.predictions[i](self.dropouts[i](g))
        return g  # Return graph embedding (external classifier head is used outside)

    def forward_with_attention(self, node_features, Adj_block, parc_type, disease_type, valid_num_nodes=None):
        """
        Run the same BrainGFM encoder path as forward(), while also returning
        graph-transformer attention maps for ROI-level attribution.

        Attention tensors have shape [B, heads, L, L], where L = N + 2 because
        BrainGFM prepends disease and parcellation tokens before the ROI nodes.
        ROI nodes therefore start at token_offset=2.
        """
        B, N, F = node_features.shape
        device = node_features.device
        if valid_num_nodes is None:
            valid_num_nodes = [N] * B

        rwse = self.compute_rwse(Adj_block, k=self.rwse_steps)
        node_features = torch.cat([node_features, rwse], dim=-1)

        padded = torch.zeros((B, N, self.max_feature_dim), device=device)
        padded[:, :, :node_features.shape[-1]] = node_features

        parc_token = self.projection_layer(self.parcellation_tokens[parc_type].expand(B, 1, -1))
        disease_type = disease_type.lower()
        if disease_type not in self.disease_embeddings:
            disease_type = 'none'
        disease_token = self.disease_proj(self.disease_embeddings[disease_type].expand(B, 1, -1))

        x_prompted, A_tilde, attn_bias_nodes = self.graph_prompt(
            node_feats_BND=padded,
            adj_BNN=Adj_block.float(),
            disease_token_B1H=disease_token,
            parc_token_B1H=parc_token,
            valid_num_nodes=valid_num_nodes
        )

        x_proj = self.projection_layer(x_prompted)
        x = torch.cat([disease_token, parc_token, x_proj], dim=1)
        Adj_block_with_tokens = self.expand_adj_block(A_tilde, num_tokens=2)

        padding_mask = torch.ones(B, N + 2, device=device).bool()
        for i, n_valid in enumerate(valid_num_nodes):
            padding_mask[i, :n_valid + 2] = False

        out = 0
        attentions = []
        for i in range(len(self.ugformer_layers)):
            h = x
            encoder = self.ugformer_layers[i]
            for layer_idx, layer in enumerate(encoder.layers):
                h, attn = layer(
                    h,
                    src_key_padding_mask=padding_mask,
                    return_attention=True
                )
                attentions.append({
                    'gnn_layer': i,
                    'transformer_layer': layer_idx,
                    'attention': attn.detach()
                })
            if encoder.norm is not None:
                h = encoder.norm(h)

            node_h = h[:, 2:, :]
            node_mask = (~padding_mask[:, 2:]).unsqueeze(-1).float()
            node_h = self.lst_gnn[i](node_h, Adj_block_with_tokens[:, 2:, 2:])
            g = (node_h * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1e-6)
            out += self.predictions[i](self.dropouts[i](g))

        attention_info = {
            'attentions': attentions,
            'padding_mask': padding_mask.detach(),
            'valid_num_nodes': valid_num_nodes,
            'num_roi_nodes': N,
            'token_offset': 2
        }
        return g, attention_info

# ===========================================
# DiseaseGraphClassifier wrapper
# ===========================================
class DiseaseGraphClassifier(nn.Module):
    def __init__(self, encoder: BrainGFM, hidden_dim=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj, parc_type, disease_type, valid_num_nodes=None):
        g = self.encoder(x, adj, parc_type, disease_type, valid_num_nodes)
        return self.classifier(g)
