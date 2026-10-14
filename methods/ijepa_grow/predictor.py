# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""Predictor I-JEPA estreito (espelho do VisionTransformerPredictor oficial, facebookresearch/ijepa)."""

from functools import partial

import torch
from timm.models.vision_transformer import Block
from torch import nn

from methods.ijepa_grow.context import gather_tokens


def sincos_2d(dim: int, grid: int) -> torch.Tensor:
    """Pos embed sincos 2D fixo (grid*grid, dim), ordem row-major; 1a metade = coluna, 2a = linha (como o oficial)."""
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float64) / (dim // 4))
    pos = torch.arange(grid, dtype=torch.float64)
    row, col = torch.meshgrid(pos, pos, indexing="ij")

    def emb_1d(p):
        out = p.flatten()[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    return torch.cat([emb_1d(col), emb_1d(row)], dim=1).float()


class Predictor(nn.Module):
    def __init__(self, ctx_dim=384, emb_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, out_dim=1280, grid=16):
        super().__init__()
        # ctx_proj (conv 1x1) ja leva o aluno a ctx_dim; outra Linear seguida dele seria redundante.
        self.predictor_embed = nn.Identity() if ctx_dim == emb_dim else nn.Linear(ctx_dim, emb_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.register_buffer("pos_embed", sincos_2d(emb_dim, grid)[None])
        self.predictor_blocks = nn.ModuleList(
            Block(emb_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(depth)
        )
        self.predictor_norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(emb_dim, out_dim)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for i, blk in enumerate(self.predictor_blocks, 1):  # fix_init_weight do oficial
            blk.attn.proj.weight.data.div_((2.0 * i) ** 0.5)
            blk.mlp.fc2.weight.data.div_((2.0 * i) ** 0.5)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ctx, ctx_idx, tgt_idx):
        """ctx (B,Nc,ctx_dim), ctx_idx (B,Nc), tgt_idx (M,B,Nt) -> (M,B,Nt,out_dim)."""
        M, B, Nt = tgt_idx.shape
        Nc = ctx.size(1)
        pos = self.pos_embed.expand(B, -1, -1)
        x = self.predictor_embed(ctx) + gather_tokens(pos, ctx_idx)
        x = x.repeat(M, 1, 1)  # (M*B,Nc,D), mesma ordem do repeat_interleave_batch oficial
        pos = pos.repeat(M, 1, 1)
        tgt = self.mask_token + gather_tokens(pos, tgt_idx.reshape(M * B, Nt))
        x = torch.cat([x, tgt], dim=1)
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_proj(self.predictor_norm(x[:, Nc:]))
        return x.view(M, B, Nt, -1)


class ConvPredictor(nn.Module):
    """Predictor do CNN-JEPA (Kalapos & Gyires-Toth, 2024, arXiv:2408.07514, Sec. III-C): grade 16x16 com o
    contexto e um [MASK] aprendivel no resto, `depth` x (depthwise 3x3 -> pointwise 1x1 -> BN -> ReLU),
    e conv 1x1 -> out_dim (o alvo aqui e o ViT, nao o proprio encoder). Sem pos embed: a conv ja sabe onde esta."""

    def __init__(self, ctx_dim=384, depth=3, out_dim=1280, grid=16):
        super().__init__()
        self.grid = grid
        self.mask_token = nn.Parameter(torch.zeros(1, 1, ctx_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(ctx_dim, ctx_dim, 3, padding=1, groups=ctx_dim, bias=False),
            nn.Conv2d(ctx_dim, ctx_dim, 1, bias=False),
            nn.BatchNorm2d(ctx_dim),
            nn.ReLU(inplace=True),
        ) for _ in range(depth)])
        self.proj = nn.Conv2d(ctx_dim, out_dim, 1)

    def forward(self, ctx, ctx_idx, tgt_idx):
        """Mesmo contrato do Predictor: ctx (B,Nc,D), ctx_idx (B,Nc), tgt_idx (M,B,Nt) -> (M,B,Nt,out_dim).
        Todo alvo ve so o contexto, entao a grade e uma so para os M blocos."""
        B, _, D = ctx.shape
        x = self.mask_token.expand(B, self.grid * self.grid, D)
        x = x.scatter(1, ctx_idx.unsqueeze(-1).expand(-1, -1, D), ctx.to(x.dtype))
        x = x.transpose(1, 2).reshape(B, D, self.grid, self.grid)
        x = self.proj(self.blocks(x)).flatten(2).transpose(1, 2)
        return torch.stack([gather_tokens(x, t) for t in tgt_idx])
