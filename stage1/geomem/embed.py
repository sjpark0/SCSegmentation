"""3D positional embedding for the geometry-aware memory (spec 3.2).

phi(p) = [p, sin(2^k pi p), cos(2^k pi p)]_{k=0..7}  (51 dims)  ->  MLP 51->128->128 (GELU)
  -> key head 128->64  (added to the reference memory tokens' positional encoding)
  -> query head 128->256 (added to the target view's feature positional encoding)
Cells without geometry use the learned vector e_none in place of the MLP output.  Both heads are
zero-initialised, so an untrained module leaves the tracker's behaviour exactly unchanged.
"""
import math

import torch
from torch import nn

N_FREQ = 8
HIDDEN = 128
KEY_DIM = 64
QUERY_DIM = 256


def fourier(p, n_freq=N_FREQ):
    """[..., 3] -> [..., 3 + 2*3*n_freq] = [p, sin(2^k pi p), cos(2^k pi p)] for k in 0..n_freq-1."""
    freqs = (2.0 ** torch.arange(n_freq, device=p.device, dtype=p.dtype)) * math.pi
    ang = p[..., None, :] * freqs[:, None]                      # [..., n_freq, 3]
    ang = ang.flatten(-2)                                       # [..., 3*n_freq], k-major
    return torch.cat([p, torch.sin(ang), torch.cos(ang)], dim=-1)


class GeoEmbed(nn.Module):
    def __init__(self, hidden=HIDDEN, key_dim=KEY_DIM, query_dim=QUERY_DIM, n_freq=N_FREQ):
        super().__init__()
        self.n_freq = n_freq
        in_dim = 3 + 2 * 3 * n_freq
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.e_none = nn.Parameter(torch.zeros(hidden))
        self.key_head = nn.Linear(hidden, key_dim)
        self.query_head = nn.Linear(hidden, query_dim)
        for head in (self.key_head, self.query_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, p, valid):
        """p [G,3], valid [G] bool -> (key [G,64], query [G,256])."""
        h = self.mlp(fourier(p, self.n_freq))
        h = torch.where(valid[:, None], h, self.e_none[None, :].expand_as(h))
        return self.key_head(h), self.query_head(h)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def grid_tensors(view_grid_dict, device):
    """pointmap.view_grid output -> (p [5184,3] float32, valid [5184] bool), row-major like the tracker's flatten(2)."""
    p = torch.as_tensor(view_grid_dict["p"], dtype=torch.float32, device=device).reshape(-1, 3)
    valid = torch.as_tensor(view_grid_dict["valid"], dtype=torch.bool, device=device).reshape(-1)
    return p, valid
