# graph_prompt.py
import torch
import torch.nn as nn

class GraphPrompt(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        max_nodes: int,
        max_feature_dim: int,
        node_mode: str = "scale",
        edge_strength: float = 0.3,
        init_std: float = 1e-2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.max_feature_dim = max_feature_dim
        self.node_mode = node_mode
        self.edge_strength = edge_strength

        self.base_node_prompt = nn.Parameter(torch.randn(1, max_nodes, max_feature_dim) * init_std)
        self.base_edge_prompt = nn.Parameter(torch.randn(1, max_nodes, max_nodes) * init_std)

        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [alpha_node, beta_node, alpha_edge, beta_edge]
        )

    @staticmethod
    def _mask_invalid_nodes_BND(tensor_BND, valid_num_nodes):
        # tensor: [B,N,D]
        B, N, _ = tensor_BND.shape
        device = tensor_BND.device
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for i, n in enumerate(valid_num_nodes):
            if n < N:
                mask[i, n:] = True
        return tensor_BND.masked_fill(mask.unsqueeze(-1), 0.0)

    @staticmethod
    def _mask_invalid_nodes_BNN(tensor_BNN, valid_num_nodes):
        # tensor: [B,N,N]
        B, N, _ = tensor_BNN.shape
        device = tensor_BNN.device
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for i, n in enumerate(valid_num_nodes):
            if n < N:
                mask[i, n:] = True
        tensor_BNN = tensor_BNN.masked_fill(mask.unsqueeze(-1), 0.0)
        tensor_BNN = tensor_BNN.masked_fill(mask.unsqueeze(1), 0.0)
        return tensor_BNN

    def forward(self, node_feats_BND, adj_BNN, disease_token_B1H, parc_token_B1H, valid_num_nodes):

        B, N, Dp = node_feats_BND.shape


        cond = torch.cat([disease_token_B1H.squeeze(1), parc_token_B1H.squeeze(1)], dim=-1)  # [B,2H]
        gate = self.cond_mlp(cond)  # [B,4]
        alpha_node, beta_node, alpha_edge, beta_edge = gate.chunk(4, dim=-1)
        alpha_node = alpha_node.view(B, 1, 1)
        beta_node  = beta_node.view(B, 1, 1)
        alpha_edge = alpha_edge.view(B, 1, 1)
        beta_edge  = beta_edge.view(B, 1, 1)

        # === Node prompt ===
        node_prompt = self.base_node_prompt[:, :N, :Dp].expand(B, -1, -1)       # [B,N,Dp]
        node_prompt = self._mask_invalid_nodes_BND(node_prompt, valid_num_nodes)

        if self.node_mode == "scale":
            scale = torch.tanh(alpha_node * node_prompt + beta_node)            # [-1,1]
            node_feats_prompted = node_feats_BND * (1.0 + scale)
        else:
            scale = torch.tanh(alpha_node * node_prompt + beta_node)
            bias  = torch.tanh(node_prompt)
            node_feats_prompted = node_feats_BND * (1.0 + scale) + 0.1 * bias

        # === Edge prompt ===
        edge_base = self.base_edge_prompt[:, :N, :N].expand(B, -1, -1)          # [B,N,N]
        edge_base = 0.5 * (edge_base + edge_base.transpose(1, 2))
        edge_base = edge_base - torch.diag_embed(torch.diagonal(edge_base, dim1=1, dim2=2))
        edge_base = self._mask_invalid_nodes_BNN(edge_base, valid_num_nodes)

        edge_logits = alpha_edge * edge_base + beta_edge
        edge_add = torch.sigmoid(edge_logits)                                    # [B,N,N]∈[0,1]

        Adj_soft = adj_BNN.float()
        A_tilde = torch.clamp(Adj_soft + self.edge_strength * edge_add * (1.0 - Adj_soft), 0.0, 1.0)
        A_tilde = 0.5 * (A_tilde + A_tilde.transpose(1, 2))
        A_tilde = A_tilde - torch.diag_embed(torch.diagonal(A_tilde, dim1=1, dim2=2))

        attn_bias_nodes = 3.0 * (edge_add - 0.5)  # ~[-1.5, 1.5]

        return node_feats_prompted, A_tilde, attn_bias_nodes
