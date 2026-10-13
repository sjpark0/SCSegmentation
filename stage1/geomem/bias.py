"""3D-distance attention bias for the memory cross-attention (GeoMem-2 spec 2.1).

For a target view's 72x72 cells (queries) and each memory view's cells (keys), the bias is
w · rbf(‖p_i − p_j‖) with 16 RBF centres uniform on [0, 2] (normalised scene units, σ = 0.125),
or the learned scalar b_none when either cell has no geometry.  The 4 pointer tokens of every
memory view sit at the END of the key axis and get bias 0.  Everything is zero-initialised, so an
untrained module leaves the attention unchanged.
"""
import torch
from torch import nn

N_RBF = 16
RBF_MAX = 2.0
RBF_SIGMA = 0.125
PTR_TOKENS = 4


def rbf(d, n=N_RBF, dmax=RBF_MAX, sigma=RBF_SIGMA):
    """[...] distances -> [..., n] Gaussian features centred on linspace(0, dmax, n)."""
    centres = torch.linspace(0.0, dmax, n, device=d.device, dtype=d.dtype)
    return torch.exp(-((d[..., None] - centres) ** 2) / (2 * sigma * sigma))


def pairwise_dist(p_q, p_m):
    return torch.cdist(p_q.float(), p_m.float())


class GeoBias(nn.Module):
    def __init__(self, n_rbf=N_RBF):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_rbf))
        self.b_none = nn.Parameter(torch.zeros(1))

    def forward(self, p_q, valid_q, mems):
        """p_q [Lq,3], valid_q [Lq]; mems: list of (p_m [Lm,3], valid_m [Lm]) -> [1,1,Lq, ΣLm + 4·len(mems)]."""
        blocks = []
        for p_m, valid_m in mems:
            d = pairwise_dist(p_q, p_m)                                   # [Lq, Lm]
            b = rbf(d) @ self.w.to(d.dtype)                               # [Lq, Lm]
            both = valid_q[:, None] & valid_m[None, :]
            blocks.append(torch.where(both, b, self.b_none.to(d.dtype).expand_as(b)))
        ptr = torch.zeros(p_q.shape[0], PTR_TOKENS * len(mems), device=p_q.device, dtype=blocks[0].dtype)
        return torch.cat(blocks + [ptr], dim=1)[None, None]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
