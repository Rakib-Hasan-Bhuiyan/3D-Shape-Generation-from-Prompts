import torch
import torch.nn as nn
import timm
from typing import Tuple

# --- Model Configuration ---
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FF = 1024
P_DROP = 0.1
GLOBAL_COND = True

# ---------------------------
# Building Blocks
# ---------------------------
class FFN(nn.Module):
    def __init__(self, d_model, dim_ff=1024, p_drop=0.0, act=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            act(),
            nn.Dropout(p_drop),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.net(x)


class PointDecoderBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_ff=1024, p_drop=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, dim_ff=dim_ff, p_drop=p_drop)

    def forward(self, q, mem):
        q2, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.ln1(q + q2)
        q2, _ = self.cross_attn(q, mem, mem, need_weights=False)
        q = self.ln2(q + q2)
        q2 = self.ffn(q)
        q = self.ln3(q + q2)
        return q


class PointTransformerDecoder(nn.Module):
    def __init__(self, num_queries=1024, d_model=256, nhead=8, num_layers=6, dim_ff=1024, 
                 p_drop=0.1, global_cond=True):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.global_cond = global_cond

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))
        self.global_proj = nn.Linear(d_model, d_model) if global_cond else None

        self.blocks = nn.ModuleList([
            PointDecoderBlock(d_model, nhead, dim_ff, p_drop) for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.to_xyz = nn.Linear(d_model, 3)

    def forward(self, mem_tokens, global_feat=None):
        B, T, D = mem_tokens.shape
        q = self.query_embed.expand(B, -1, -1)
        if self.global_cond and global_feat is not None:
            g = self.global_proj(global_feat)
            q = q + g.unsqueeze(1)

        for blk in self.blocks:
            q = blk(q, mem_tokens)

        q = self.head(q)  # (B, Q, D)

        xyz = self.to_xyz(q)
        return xyz


class ImageEncoderViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, freeze=True, out_dim=256, attn_heads=4):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.vit_dim = self.vit.num_features
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.token_attn = nn.MultiheadAttention(self.vit_dim, attn_heads, batch_first=True)
        self.ln_tokens = nn.LayerNorm(self.vit_dim)

        self.pool_q = nn.Parameter(torch.randn(1, 1, self.vit_dim))
        self.pool_attn = nn.MultiheadAttention(self.vit_dim, attn_heads, batch_first=True)
        self.ln_global = nn.LayerNorm(self.vit_dim)

        self.mem_proj = nn.Linear(self.vit_dim, out_dim)
        self.glob_proj = nn.Linear(self.vit_dim, out_dim)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.vit.forward_features(images)  # (B,T,Dv)
        x, _ = self.token_attn(x, x, x)
        x = self.ln_tokens(x)

        B = x.size(0)
        q = self.pool_q.expand(B, -1, -1)
        g, _ = self.pool_attn(q, x, x)
        g = self.ln_global(g.squeeze(1))

        mem = self.mem_proj(x)
        glob = self.glob_proj(g)
        return mem, glob


class Image2PCgen(nn.Module):
    def __init__(self, num_points=1024, vit_name='vit_base_patch16_224', pretrained=True, freeze_vit=True,
                 enc_attn_heads=4, d_model=256, nhead=8, num_layers=6, dim_ff=1024, p_drop=0.1,
                 global_cond=True):
        super().__init__()

        self.encoder = ImageEncoderViT(vit_name, pretrained, freeze_vit, out_dim=d_model, attn_heads=enc_attn_heads)
        self.decoder = PointTransformerDecoder(
            num_queries=num_points, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_ff=dim_ff, p_drop=p_drop,
            global_cond=global_cond
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        mem, g = self.encoder(images)
        points = self.decoder(mem, g)
        return points  # (B, N, 3)