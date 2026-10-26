# model_3dvit.py
"""
3D Vision Transformer for medical image classification.
Contains both custom lightweight ViT and official 3DINO ViT-Large with BlockChunk.
"""

import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

# ============================================================
# Custom lightweight 3D ViT components (for tiny/small/base/micro)
# ============================================================

class PatchEmbed3D(nn.Module):
    def __init__(self, volume_size=126, patch_size=14, in_channels=1, embed_dim=768, norm_layer=None):
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = volume_size // patch_size
        self.num_patches = self.grid_size ** 3
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D == self.volume_size and H == self.volume_size and W == self.volume_size
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionTransformer3D(nn.Module):
    def __init__(self,
                 volume_size=126,
                 patch_size=14,
                 in_channels=1,
                 num_classes=2,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 norm_layer=None,
                 act_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        if representation_size is None:
            self.pre_logits = nn.Identity()
            self.num_features = embed_dim
        else:
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
            self.num_features = representation_size

        self.patch_embed = PatchEmbed3D(
            volume_size=volume_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, norm_layer=None)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ============================================================
# Official 3DINO components (with BlockChunk) for exact weight loading
# ============================================================

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Mlp3DINO(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention3DINO(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x


class Block3DINO(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_bias=True, ffn_bias=True,
                 drop=0., attn_drop=0., init_values=None, drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_class=Attention3DINO, ffn_layer=Mlp3DINO):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                               attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                             drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class PatchEmbed3DINO(nn.Module):
    def __init__(self, volume_size=112, patch_size=16, in_channels=1, embed_dim=1024, norm_layer=None):
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = volume_size // patch_size
        self.num_patches = self.grid_size ** 3
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D == self.volume_size and H == self.volume_size and W == self.volume_size
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class DinoVisionTransformer3d(nn.Module):
    """
    3DINO ViT-Large with official 24-layer BlockChunk structure.
    Compatible with 3DINO pretrained weights (teacher checkpoint).
    """
    def __init__(
        self,
        volume_size=112,
        patch_size=16,
        in_channels=1,
        num_classes=2,
        embed_dim=1024,
        depth=24,                # 官方 ViT-Large 为 24 层
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop_path_rate=0.0,
        init_values=1e-5,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3DINO(
            volume_size=volume_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=None,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # Build blocks exactly like the official code using block_chunks=4
        block_chunks = 4
        chunksize = depth // block_chunks   # 24 // 4 = 6
        self.chunked_blocks = True
        self.n_blocks = depth

        # Create all 24 blocks
        blocks_list = [
            Block3DINO(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=0.0,
                attn_drop=0.0,
                init_values=init_values,
                drop_path=drop_path_rate * i / (depth - 1),   # linear drop path
                act_layer=act_layer,
                norm_layer=norm_layer,
                attn_class=Attention3DINO,
                ffn_layer=Mlp3DINO,
            )
            for i in range(depth)
        ]

        # Group into chunks with identity padding for FSDP compatibility (same as official)
        chunked_blocks = []
        for i in range(0, depth, chunksize):
            # i 是起始索引，前面填充 i 个 Identity，然后加上本 chunk 的 6 个真实 Block
            chunk = [nn.Identity()] * i + blocks_list[i : i + chunksize]
            chunked_blocks.append(BlockChunk(chunk))

        self.blocks = nn.ModuleList(chunked_blocks)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # Weight initialization (identical to official)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ============================================================
# Factory functions
# ============================================================

def vit_3d_tiny(volume_size=126, patch_size=14, in_channels=1, num_classes=2, **kwargs):
    embed_dim = kwargs.pop('embed_dim', 192)
    depth = kwargs.pop('depth', 12)
    num_heads = kwargs.pop('num_heads', 3)
    mlp_ratio = kwargs.pop('mlp_ratio', 4.0)
    representation_size = kwargs.pop('representation_size', embed_dim)
    drop_ratio = kwargs.pop('drop_ratio', 0.1)
    attn_drop_ratio = kwargs.pop('attn_drop_ratio', 0.0)
    qkv_bias = kwargs.pop('qkv_bias', True)
    model = VisionTransformer3D(
        volume_size=volume_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias, representation_size=representation_size,
        drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, **kwargs)
    return model


def vit_3d_small(volume_size=126, patch_size=14, in_channels=1, num_classes=2, **kwargs):
    embed_dim = kwargs.pop('embed_dim', 384)
    depth = kwargs.pop('depth', 12)
    num_heads = kwargs.pop('num_heads', 6)
    mlp_ratio = kwargs.pop('mlp_ratio', 4.0)
    representation_size = kwargs.pop('representation_size', embed_dim)
    drop_ratio = kwargs.pop('drop_ratio', 0.1)
    attn_drop_ratio = kwargs.pop('attn_drop_ratio', 0.0)
    qkv_bias = kwargs.pop('qkv_bias', True)
    model = VisionTransformer3D(
        volume_size=volume_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias, representation_size=representation_size,
        drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, **kwargs)
    return model


def vit_3d_base(volume_size=126, patch_size=14, in_channels=1, num_classes=2, **kwargs):
    embed_dim = kwargs.pop('embed_dim', 768)
    depth = kwargs.pop('depth', 12)
    num_heads = kwargs.pop('num_heads', 12)
    mlp_ratio = kwargs.pop('mlp_ratio', 4.0)
    representation_size = kwargs.pop('representation_size', embed_dim)
    drop_ratio = kwargs.pop('drop_ratio', 0.1)
    attn_drop_ratio = kwargs.pop('attn_drop_ratio', 0.0)
    qkv_bias = kwargs.pop('qkv_bias', True)
    model = VisionTransformer3D(
        volume_size=volume_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias, representation_size=representation_size,
        drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, **kwargs)
    return model


def vit_3d_micro(volume_size=126, patch_size=6, in_channels=1, num_classes=2, **kwargs):
    embed_dim = kwargs.pop('embed_dim', 48)
    depth = kwargs.pop('depth', 3)
    num_heads = kwargs.pop('num_heads', 3)
    mlp_ratio = kwargs.pop('mlp_ratio', 1.5)
    representation_size = kwargs.pop('representation_size', None)
    drop_ratio = kwargs.pop('drop_ratio', 0.2)
    attn_drop_ratio = kwargs.pop('attn_drop_ratio', 0.0)
    qkv_bias = kwargs.pop('qkv_bias', True)
    model = VisionTransformer3D(
        volume_size=volume_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias, representation_size=representation_size,
        drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, **kwargs)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[vit_3d_micro] 参数量: {total_params:,}")
    return model


def vit_3dino_large(volume_size=112, patch_size=16, in_channels=1, num_classes=2, **kwargs):
    """3DINO-ViT-Large using official DinoVisionTransformer3d (24 layers)."""
    model = DinoVisionTransformer3d(
        volume_size=volume_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop_path_rate=0.0,
        init_values=1e-5,
    )
    return model
