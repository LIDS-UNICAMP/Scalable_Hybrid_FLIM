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
"""VICRegL (Bardes, Ponce, LeCun, 2210.01571) on encoder maps, no expander, no labels.

Follows the official code: feature-space matching keeps all pairs
(`--l2_all_matches 1`); only position matching keeps the top-gamma pairs.
"""

import torch
import torch.nn.functional as F


def _vicreg(x, y, inv_coeff, var_coeff, cov_coeff, paper_cov):
    """x, y: [B, M, C]. Variance and covariance run over the batch, per slot M."""
    inv = F.mse_loss(x, y)
    var = cov = 0.0
    n, _, c = x.shape
    for v in (x, y):
        v = v - v.mean(0)
        diag = v.pow(2).sum(0) / (n - 1)
        var = var + F.relu(1.0 - torch.sqrt(diag + 1e-4)).mean() / 2
        # ||V^T V||_F == ||V V^T||_F: a BxB Gram avoids a CxC matrix per slot.
        gram = torch.einsum("bmc,dmc->mbd", v, v) / (n - 1)
        off = gram.pow(2).sum((1, 2)) - diag.pow(2).sum(-1)
        cov = cov + off.mean() / (c if paper_cov else c * (c - 1)) / 2
    return inv_coeff * inv + var_coeff * var + cov_coeff * cov


def _pairs(a, b, dist, num_matches=None):
    """Nearest b for each token of a, keeping the num_matches closest pairs."""
    best, nn = dist.min(-1)
    if num_matches is None:
        return a, b.gather(1, nn[..., None].expand(-1, -1, b.size(-1)))
    keep = best.topk(num_matches, largest=False).indices.sort(-1).values
    nn = nn.gather(1, keep)
    return (a.gather(1, keep[..., None].expand(-1, -1, a.size(-1))),
            b.gather(1, nn[..., None].expand(-1, -1, b.size(-1))))


def vicregl_loss(maps_1, maps_2, loc_1=None, loc_2=None, alpha=0.75, num_matches=(20, 20),
                 inv_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, paper_cov=False, max_grid=0):
    """maps: [B, C, H, W] before pooling. loc: [B, H, W, 2] image coordinates of each token.

    loc=None means the maps are already aligned token by token (flip or rotation
    undone by the caller): position matching is then the identity, on all tokens.
    Top-gamma would only break ties there, always at the same corner.

    paper_cov=True divides covariance by C as the paper does; with C >> B its
    noise floor (C-1)/(B-1) outweighs the variance term. Default divides by C(C-1).

    max_grid > 0 pools the local term down to max_grid x max_grid (loc=None only): cdist is
    [B, HW, HW], which at the 99x99 map of L1 is 12 GB by itself. The paper works on 7x7.
    """
    w = (inv_coeff, var_coeff, cov_coeff, paper_cov)
    glob = _vicreg(maps_1.mean((2, 3))[:, None], maps_2.mean((2, 3))[:, None], *w)

    if max_grid and maps_1.shape[-1] > max_grid:
        maps_1, maps_2 = (F.adaptive_avg_pool2d(m, max_grid) for m in (maps_1, maps_2))
    a, b = maps_1.flatten(2).transpose(1, 2), maps_2.flatten(2).transpose(1, 2)
    with torch.no_grad():
        d_feat = torch.cdist(a, b)
    local = (_vicreg(*_pairs(a, b, d_feat), *w) + _vicreg(*_pairs(b, a, d_feat.transpose(1, 2)), *w)) / 2

    if loc_1 is None:
        local = local + _vicreg(a, b, *w)
    else:
        d_loc = torch.cdist(loc_1.flatten(1, 2).float(), loc_2.flatten(1, 2).float())
        local = local + (_vicreg(*_pairs(a, b, d_loc, num_matches[0]), *w)
                         + _vicreg(*_pairs(b, a, d_loc.transpose(1, 2), num_matches[1]), *w)) / 2
    return alpha * glob + (1 - alpha) * local


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H = 32, 1280, 8
    iso = torch.randn(B, C, H, H)
    iso_2 = iso + 0.1 * torch.randn_like(iso)
    const = torch.ones(B, C, H, H)
    rank1 = torch.randn(B, 1, H, H).expand(-1, C, -1, -1).contiguous()
    grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij"), -1).float()
    loc = grid.expand(B, -1, -1, -1)
    for name, kw in (("posicao", dict(loc_1=loc, loc_2=loc + 0.3)), ("alinhado", {})):
        for paper in (False, True):
            r = {k: vicregl_loss(m, m2, paper_cov=paper, **kw).item() for k, (m, m2) in
                 dict(iso=(iso, iso_2), const=(const, const), rank1=(rank1, rank1)).items()}
            print(name, "cov/C" if paper else "cov/C(C-1)", {k: round(v, 3) for k, v in r.items()})
            if not paper:
                assert r["const"] > r["iso"] and r["rank1"] > r["iso"]
    x = iso.clone().requires_grad_()
    vicregl_loss(x, iso_2).backward()
    assert x.grad.isfinite().all()
    print("ok")
