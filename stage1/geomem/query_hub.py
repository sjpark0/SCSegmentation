"""Object-query hub (GeoQuery spec 2.3): M object queries read every view's tokens with a 3D
object-cloud distance bias, talk to each other, and are projected into a SAM prompt token.

Everything that changes the queries is behind a scalar gate initialised to 0 (`x + g·block(norm(x))`)
and `proj0`, `proj_q`, `type_embed` are zero-initialised, so an untrained hub returns the object pointers
unchanged and a zero prompt token.  Attention is explicit (fp32 softmax) so the attention mass per
view is available for the hub statistics.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from geomem.bias import N_RBF, rbf


class ObjBias(nn.Module):
    """Per-(object, token) additive logit from the token-to-cloud distance; `b_none` where the distance is inf."""

    def __init__(self, n_rbf=N_RBF):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_rbf))
        self.b_none = nn.Parameter(torch.zeros(1))

    def forward(self, dist):
        finite = torch.isfinite(dist)
        d = torch.where(finite, dist, torch.zeros_like(dist))
        b = rbf(d.float()) @ self.w.float()
        return torch.where(finite, b, self.b_none.float().expand_as(b))


def _attention(q, k, v, heads, bias=None):
    """q [M,D], k/v [L,D], bias [M,L] or None -> (out [M,D], attn [M,L] head-averaged, fp32)."""
    M, D = q.shape
    dh = D // heads
    qh = q.float().view(M, heads, dh).transpose(0, 1)                     # [h,M,dh]
    kh = k.float().view(-1, heads, dh).transpose(0, 1)                    # [h,L,dh]
    vh = v.float().view(-1, heads, dh).transpose(0, 1)
    logits = qh @ kh.transpose(1, 2) / math.sqrt(dh)                     # [h,M,L]
    if bias is not None:
        logits = logits + bias.float()[None]
    attn = torch.softmax(logits, dim=-1)
    out = (attn @ vh).transpose(0, 1).reshape(M, D)
    return out.to(q.dtype), attn.mean(0)


class HubLayer(nn.Module):
    def __init__(self, dim, heads, ffn):
        super().__init__()
        self.heads = heads
        self.norm_cross, self.norm_self, self.norm_ffn = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(dim, dim) for _ in range(4))
        self.sq_proj, self.sk_proj, self.sv_proj, self.so_proj = (nn.Linear(dim, dim) for _ in range(4))
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))
        self.g_cross, self.g_self, self.g_ffn = (nn.Parameter(torch.zeros(1)) for _ in range(3))

    def forward(self, x, k, v, bias):
        h = self.norm_cross(x)
        out, attn = _attention(self.q_proj(h), k, v, self.heads, bias)
        x = x + self.g_cross * self.o_proj(out)
        h = self.norm_self(x)
        out, _ = _attention(self.sq_proj(h), self.sk_proj(h), self.sv_proj(h), self.heads)
        x = x + self.g_self * self.so_proj(out)
        x = x + self.g_ffn * self.ffn(self.norm_ffn(x))
        return x, attn


class QueryHub(nn.Module):
    def __init__(self, dim=256, layers=2, heads=8, ffn=1024, use_bias=True, ptr=False, g_cross_init=0.0):
        """`ptr=True` adds `ptr_proj` (zero-init) for injecting the query as a delta on the reference view's 4
        pointer tokens (memory-attention path); `g_cross_init` opens the cross-attention gates from the start so the
        attention can become object-specific (diagnosis 2026-10-08: with g_cross=0 the cross output stays a constant)."""
        super().__init__()
        self.dim = dim
        self.proj0 = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(HubLayer(dim, heads, ffn) for _ in range(layers))
        self.obj_bias = ObjBias() if use_bias else None
        self.proj_q = nn.Linear(dim, dim)
        self.type_embed = nn.Parameter(torch.zeros(dim))
        lins = [self.proj0, self.proj_q]
        if ptr:
            self.ptr_proj = nn.Linear(dim, dim)
            lins.append(self.ptr_proj)
        for lin in lins:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        with torch.no_grad():
            for layer in self.layers:
                layer.g_cross.fill_(float(g_cross_init))

    def forward(self, obj_ptr, tokens, dist, view_ids):
        """obj_ptr [M,D], tokens [L,D], dist [M,L] or None, view_ids [L] in 0..N-1 -> (q [M,D], mass [layers,M,N])."""
        n_views = int(view_ids.max().item()) + 1 if view_ids.numel() else 0
        bias = self.obj_bias(dist) if (self.obj_bias is not None and dist is not None) else None
        x = obj_ptr + self.proj0(obj_ptr)
        masses = []
        for layer in self.layers:
            x, attn = layer(x, layer.k_proj(tokens), layer.v_proj(tokens), bias)
            mass = torch.zeros(attn.shape[0], n_views, device=attn.device, dtype=attn.dtype)
            masses.append(mass.index_add(1, view_ids, attn))
        return x, torch.stack(masses)

    def prompt_token(self, q):
        return self.proj_q(q) + self.type_embed

    def ptr_delta(self, q):
        """q [M,256] -> delta for the 4 pointer tokens of mem_dim 64: [4, M, 64] (same split as MemoryView packing)."""
        M = q.shape[0]
        return self.ptr_proj(q).view(M, 4, self.dim // 4).permute(1, 0, 2)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
