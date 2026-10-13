"""GeoFuse spec 2.2: ViewFusion — per-cell cross-view attention over [self, aligned view 1..N] with a zero-init gate.

Every target cell attends from its own backbone feature (query) to the token set {its own feature, the N
views' features aligned to that cell (geomem.align)}; invalid aligned tokens are masked.  The self token is
always valid, so no cell ever has an empty key set.  F' = F + g * o_proj(attn), g = 0 at init, so an
untrained module is the identity byte for byte.  Statistics (no grad, head-averaged): the attention mass
on the self token, the normalised entropy of the mass over the N views (cells with >= 1 valid view) and
the mean number of valid views per cell — the "did it open?" readouts GeoQuery taught us to record.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewFusion(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads = dim, heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, F_t, W, valid):
        """F_t [L,dim], W [N,L,dim], valid [N,L] bool -> (F' [L,dim], stats dict)."""
        L, dim = F_t.shape
        N = W.shape[0]
        H, d = self.heads, dim // self.heads
        tokens = torch.cat([F_t[None], W.to(F_t.dtype)], dim=0)                    # [N+1, L, dim]
        mask = torch.cat([torch.ones(1, L, dtype=torch.bool, device=F_t.device), valid.to(F_t.device)], dim=0)  # [N+1, L]
        q = self.q_proj(F_t).view(L, H, d)                                          # [L,H,d]
        k = self.k_proj(tokens).view(N + 1, L, H, d)                                # [N+1,L,H,d]
        v = self.v_proj(tokens).view(N + 1, L, H, d)
        logits = torch.einsum("lhd,nlhd->lhn", q, k) / (d ** 0.5)                   # [L,H,N+1]
        logits = logits.masked_fill(~mask.T[:, None, :], float("-inf"))
        attn = logits.softmax(dim=-1)                                               # [L,H,N+1]
        out = torch.einsum("lhn,nlhd->lhd", attn, v).reshape(L, dim)
        F_out = F_t + self.g * self.o_proj(out)
        with torch.no_grad():
            a = attn.float().mean(dim=1)                                            # [L,N+1] head-averaged
            self_mass = a[:, 0]
            views = a[:, 1:]
            n_valid = valid.to(F_t.device).sum(0)                                   # [L]
            has = n_valid > 0
            if bool(has.any()):
                p = views[has] / views[has].sum(-1, keepdim=True).clamp(min=1e-12)
                ent = -(p * (p.clamp(min=1e-12)).log()).sum(-1)
                denom = torch.log(n_valid[has].float().clamp(min=2.0))             # a single valid view has entropy 0
                entropy = float((ent / denom).mean())
            else:
                entropy = 0.0
            stats = {"self_mass": float(self_mass.mean()), "view_entropy": entropy,
                     "valid_views_mean": float(n_valid.float().mean())}
        return F_out, stats
